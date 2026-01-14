#include "yaml-cpp/yaml.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <datetime/calendar.hpp>
#include <limits>
#include <stdexcept>

/* Eigen v3 */
#include "eigen3/Eigen/Eigen"

/* headers for integrator, repo: integrator */
#include "orbit_integrator/integration_parameters.hpp"

/* headers for DORIS Rinex, repo librnx */
#include "rnx/doris_rinex.hpp"

/* headers for DPOD Sinex */
#include "sinex/dpod.hpp"

/* headers for troposphere (VMF3) */
#include "rwatmo/vmf3.hpp"

/* astronomy */
#include "iers/iau.hpp"

/* iers */
#include "iers/earth_rotation.hpp"
#include "iers/gravity.hpp"
#include "iers/relativity.hpp"

/* satellites and systems */
#include "sysnsats/doris.hpp"
#include "sysnsats/doris_ground_segment.hpp"
#include "sysnsats/drag.hpp"
#include "sysnsats/occultation.hpp"
#include "sysnsats/satellite.hpp"
#include "sysnsats/satellite_payload.hpp"
#include "sysnsats/srp.hpp"

/* integrator */
#include "orbit_integrator/dop853.hpp"

/* sp3 for initial conditions */
#include "sp3/sp3.hpp"

/* geodesy (transformations+units) */
#include "geodesy/transformations.hpp"
#include "geodesy/units.hpp"

constexpr const double GM_Moon = 4902.800076e9;
constexpr const double GM_Sun = 132712440040.944e9;

using DorisObservationType = dso::SatelliteSystemObservationType<
    dso::SATELLITE_SYSTEM::DORIS>::ObservationType;
using DorisSystem = dso::SatelliteSystemTraits<dso::SATELLITE_SYSTEM::DORIS>;

/* if two consecutive observations (same beacon) are more than
 * _MIN_DIFF_FOR_NEW_ARC_ seconds apart, we introduce a new arc.
 */
constexpr const double _MIN_DIFF_FOR_NEW_ARC_ = 5. * 60. * 60.;
/* if two consecutive observations (same beacon) are more than
 * _MIN_DIFF_FOR_DOPPLER_ seconds apart, we re-initialize the Doppler count.
 */
constexpr const double _MIN_DIFF_FOR_DOPPLER_ = 10.;

/* minimum elevation cut-off angle in [degrees] */
constexpr const double MIN_ELEVATION_DEG = 7.;

/* a-priori std. deviation for a Doppler observation in [m/sec] */
constexpr const double OBS_SIGMA_0 = 1e-2;

/* set these (tune later) */
constexpr double q_drag = 1e-10;
constexpr double q_srp = 1e-10;
constexpr int NSTM = 6 * 6;  /* STM matrix */
constexpr int NSENS = 6 * 2; /* Sensitivity Matrix, includes Cd and Cr */

/* Assuming that the STM matrix is in "vectorized" form within the y vector
 * (which should have size 6+6x6 rows), extract the STM from the (column)vector
 * and return it as a 6x6 matrix
 */
Eigen::Matrix<double, 6, 6> vec2stm(const Eigen::VectorXd &y) noexcept {
  Eigen::Matrix<double, 6, 6> F;
  for (int i = 0; i < 6; i++) {
    F.block<6, 1>(0, i) = y.segment<6>(6 + i * 6);
  }
  return F;
}

struct BeaconObs {
  const dso::doris_rnx::Beacon
      *bcn_ptr; /* ptr to a RINEX instance's internal vector */
  Eigen::Vector3d rsat_gcrf;
  Eigen::Vector3d rsta_gcrf;
  dso::MjdEpoch tai;
  dso::vmf3::Vmf3Result vmf3;
  double az, el;
  double Lif, L2GHz, L400MHz;
  double fen, frt;
  int pass_nr{0};

  /* Compute tropospheric delay [m] using the instance's vmf3 member. 
   * Optional, we can pass ZWD [m] else the function will use whatever is 
   * stored in vmf3.zwd (probably a-priori value).
   */
  double
  dTropo(double zwd = std::numeric_limits<double>::min()) const noexcept {
    zwd = (zwd == std::numeric_limits<double>::min()) ? vmf3.zwd() : zwd;
    return vmf3.zhd() * vmf3.mfh() + zwd * vmf3.mfw();
  }

  void replace_data(const Eigen::Vector3d &_rsat_gcrf,
                    const Eigen::Vector3d &_rsta_gcrf,
                    const dso::MjdEpoch &_tai, double _az, double _el,
                    double _Lif, double _L2GHz, double _L400MHz, double _fen,
                    double _frt, const dso::vmf3::Vmf3Result &_vmf3) noexcept {
    this->rsat_gcrf = _rsat_gcrf;
    this->rsta_gcrf = _rsta_gcrf;
    this->tai = _tai;
    this->az = _az;
    this->el = _el;
    this->Lif = _Lif;
    this->L2GHz = _L2GHz;
    this->L400MHz = _L400MHz;
    this->fen = _fen;
    this->frt = _frt;
    this->vmf3 = _vmf3;
  }

  /* Equation (13) from Lemoine et al, 2010 */
  double vobserved(const BeaconObs &next) const noexcept {
    constexpr const double C = iers2010::C;
    const double dtau =
        next.tai.diff<dso::DateTimeDifferenceType::FractionalSeconds>(this->tai)
            .seconds();                       // [sec]
    const double Ndop = next.Lif - this->Lif; // [cycles]]
    const double mfrt = (this->frt + next.frt) / 2.;
    printf("\tObserved value from dtau=%.9f, Ndop=%.3f, mfrt=%.3f, fen=%.3f, "
           "Dtropo=(%.3f-%.3f)/%.9f\n",
           dtau, Ndop, mfrt, fen, next.dTropo(), this->dTropo(), dtau);
    const double Vm = -(C / fen) * (fen - mfrt - Ndop / dtau) -
                      (next.dTropo() - this->dTropo()) / dtau; // [m/sec]
    return Vm;                                                 // [m/sec]
  }

  /* Equation (13) from Lemoine et al, 2010
   * Note that the calling instance is at the "point of linearization"
   */
  double vcomputed(const BeaconObs &next, const Eigen::MatrixXd &STM,
                   double dfe, Eigen::VectorXd &H) const noexcept {
    constexpr const double C = iers2010::C;
    const double dtau =
        next.tai.diff<dso::DateTimeDifferenceType::FractionalSeconds>(this->tai)
            .seconds();                       // [sec]
    const double Ndop = next.Lif - this->Lif; // [cycles]]
    const double mfrt = (this->frt + next.frt) / 2.;

    const Eigen::Vector3d u1 = this->rsat_gcrf - this->rsta_gcrf;
    const double rho1 = u1.norm();

    const Eigen::Vector3d u2 = next.rsat_gcrf - next.rsta_gcrf;
    const double rho2 = u2.norm();

    const double Vc =
        (rho2 - rho1) / dtau - (C / fen) * (Ndop / dtau + mfrt) * dfe;

    /* partials matrix, H = [dh/dr, dh/dv, dh/ddfe] */
    Eigen::VectorXd b1 = Eigen::VectorXd(6);
    b1.segment<3>(0) = u1 / rho1;
    b1.segment<3>(3) = Eigen::Vector3d::Zero();
    Eigen::VectorXd b2 = Eigen::VectorXd(6);
    b2.segment<3>(0) = u2 / rho2;
    b2.segment<3>(3) = Eigen::Vector3d::Zero();
    H.setZero();
    /* STM is t1 -> t2, aka
    δx2 = Φ*δx1 => δx1 = Φ^(-1) * δx2
    δρ1 = b1^(T)* δx1 = b1^(T) * Φ^(-1) * δx2
    δg = (1/Δt) * (δρ2 - δρ1) = (1/Δt) * [b2^(T) - b1^(T) * Φ^(-1)] δx2
    Instead of inverting, we could solve for:
    Φ^(T) * z = b1 => z = Φ^(-T) * b1
    */
    Eigen::Matrix<double, 6, 1> z = STM.transpose().fullPivLu().solve(b1);
    H.segment<6>(0) = (b2 - z) / dtau;
    H(6) = -(C / fen) * (Ndop / dtau + mfrt);

    return Vc; // [m/sec]
  }

}; /* BeaconObs */

struct Kalman {
  /* number of estimated parameters:
   * 6                  -> satellite state vector (GCRF)
   * 1 (#6)             -> drag scalling coefficient, Cd
   * 1 (#7)             -> srp scalling coefficient, Cr
   * 1 * num of beacons -> (running) relative frequency offset per beacon
   *                       starting at index #8
   */
  const int _rfo_start_index = 8;
  const int _Cd_index = 6;
  const int _Cr_index = 7;

  dso::MjdEpoch tai_;
  Eigen::VectorXd x_; /* estimates */
  Eigen::MatrixXd P_; /* P matrix */

  int num_params() const noexcept { return x_.rows(); }

  Kalman(int numParams, const dso::MjdEpoch &_tai = dso::MjdEpoch{}) noexcept
      : tai_(_tai), x_(Eigen::VectorXd::Zero(numParams)),
        P_(Eigen::MatrixXd::Identity(numParams, numParams)) {}

  auto tai() const noexcept { return tai_; }
  auto &tai() noexcept { return tai_; }
  auto state_vector_gcrf() const noexcept { return x_.segment<6>(0); }

  const double &rfo(int beacon_index) const noexcept {
    return x_(_rfo_start_index + beacon_index);
  }

  double &rfo(int beacon_index) noexcept {
    return x_(_rfo_start_index + beacon_index);
  }

  int idx_sd() const noexcept { return _Cd_index; }
  int idx_ss() const noexcept { return _Cr_index; }

  double drag_scale() const noexcept { return x_(idx_sd()); }
  double &drag_scale() noexcept { return x_(idx_sd()); }
  double srp_scale() const noexcept { return x_(idx_ss()); }
  double &srp_scale() noexcept { return x_(idx_ss()); }

  Eigen::Vector3d position() const noexcept { return x_.segment<3>(0); }

  void reset_rfo(int beacon_index, double val0, double sigma0) noexcept {
    const int i = beacon_index + _rfo_start_index;
    rfo(beacon_index) = val0;
    P_.row(i).setZero();
    P_.col(i).setZero();
    P_(i, i) = sigma0 * sigma0;
  }
  void reset_Cd(double val0, double sigma0) noexcept {
    drag_scale() = val0;
    P_.row(idx_sd()).setZero();
    P_.col(idx_sd()).setZero();
    P_(idx_sd(), idx_sd()) = sigma0 * sigma0;
  }
  void reset_Cr(double val0, double sigma0) noexcept {
    srp_scale() = val0;
    P_.row(idx_ss()).setZero();
    P_.col(idx_ss()).setZero();
    P_(idx_ss(), idx_ss()) = sigma0 * sigma0;
  }

  void time_update(const dso::MjdEpoch &tai, const Eigen::VectorXd &ystm,
                   double q_drag, // [1/s], random walk spectral density
                   double q_srp, Eigen::Matrix<double, 6, 6> &STM) {
    /* keep parameter states as-is, update only orbit part */
    x_.segment<6>(0) = ystm.segment<6>(0);

    const int N = num_params();
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(N, N);

    /* orbit block STM */
    for (int i = 0; i < 6; i++)
      F.block<6, 1>(0, i) = ystm.segment<6>(6 + 6 * i);

    /* orbit <- scale coupling */
    F.block<6, 1>(0, idx_sd()) = ystm.segment<6>(6 + NSTM + 0 * 6);
    F.block<6, 1>(0, idx_ss()) = ystm.segment<6>(6 + NSTM + 1 * 6);

    /* covariance propagation */
    P_ = F * P_ * F.transpose();

    /* add process noise for random-walk scale factors: Var += q * dt */
    const double dt_sec =
        tai.diff<dso::DateTimeDifferenceType::FractionalSeconds>(this->tai())
            .seconds();
    P_(idx_sd(), idx_sd()) += q_drag * dt_sec;
    P_(idx_ss(), idx_ss()) += q_srp * dt_sec;

    P_ = 0.5 * (P_ + P_.transpose());
    tai_ = tai;

    /* return STM, 6x6 */
    STM = F.block<6, 6>(0, 0);
  }

  /* Observation update step (one observation at a time), with:
   * residual = z - g (scalar equation)
   *
   * @param[in] z observed value (scalar)
   * @param[in] g computed value (scalar)
   * @param[in] sigma observation std. deviation (scalar)
   * @param[in] H partials of observation w.r.t the estimated parameters, i.e.
   * d(obs_equation)/dx. Single col, rows should match the columns of x
   * vector.
   */
  void observation_update(double z, double g, double sigma,
                          const Eigen::VectorXd &H) noexcept {
    double inv_w = sigma * sigma;
    /* kalman gain */
    const auto K = P_ * H / (inv_w + H.dot(P_ * H));
    /* state update (note that y is a scalar y <- z - g = Y(k) - G(X,t) */
    x_ = x_ + K * (z - g);
    const int N = x_.rows();
    /* covariance update (Joseph variant) */
    auto KWm1Kt = (K * sigma) * (K * sigma).transpose();
    auto ImKG = Eigen::MatrixXd::Identity(N, N) - K * H.transpose();
    P_ = ImKG * P_ * ImKG.transpose() + KWm1Kt;
  }
}; /* Kalman */

/* compute satellite acceleration */
int satacc(double tsec, Eigen::Ref<const Eigen::VectorXd> y0,
           dso::IntegrationParameters *params,
           Eigen::Ref<Eigen::VectorXd> yp) noexcept;

/* Compute relevant quaternions for the ITRS/GCRS transformation */
int prep_c2i(const dso::MjdEpoch &tai, dso::EopSeries &eops,
             Eigen::Quaterniond &q_c2tirs, Eigen::Quaterniond &q_tirs2i,
             double *fargs, dso::EopRecord &eopr) noexcept;

Eigen::Matrix<double, 6, 1>
gcrf2itrf(const dso::MjdEpoch &tai,
          const Eigen::Matrix<double, 6, 1> &state_gcrf,
          dso::IntegrationParameters *params) noexcept;

Eigen::Matrix<double, 3, 1>
gcrf2itrf(const dso::MjdEpoch &tai,
          const Eigen::Matrix<double, 3, 1> &state_gcrf,
          dso::IntegrationParameters *params) noexcept;

Eigen::Matrix<double, 6, 1>
itrf2gcrf(const dso::MjdEpoch &tai,
          const Eigen::Matrix<double, 6, 1> &state_itrf,
          dso::IntegrationParameters *params) noexcept;

Eigen::Matrix<double, 3, 1>
itrf2gcrf(const dso::MjdEpoch &tai,
          const Eigen::Matrix<double, 3, 1> &state_itrf,
          dso::IntegrationParameters *params) noexcept;

int main(int argc, char *argv[]) {
  // check input
  if (argc != 7) {
    fprintf(stderr,
            "USAGE: %s [YAML CONFIG] [RINEX] [DPOD SINEX] [DPOD FREQ CORR] "
            "[VMF3 DATA GRID] [SP3c]\n",
            argv[0]);
    return 1;
  }

  try {
    /* create a RINEX instance */
    dso::DorisObsRinex rnx(argv[2]);
    const auto t_start = rnx.time_first_obs();

    /* Station/Beacon coordinates
     * -----------------------------------------------------------------------
     * Get beacon coordinates from sinex file and extrapolate to RINEX ref.
     * time Result coordinates per beacon are stored in the beaconCrdVec
     * vector. Note that these position vectors are w.r.t the beacon/antenna
     * reference point. When in actual processing, this has to be changed, if
     * we are considering iono-free analysis.
     */
    std::vector<dso::Sinex::SiteCoordinateResults> beaconCrdVec;
    if (dso::dpod_extrapolate(t_start, rnx.beacons_4charids(), beaconCrdVec,
                              argv[3], argv[4])) {
      fprintf(
          stderr,
          "ERROR: Failed extrapolating SINEX coordinates, SINEX file is %s\n",
          argv[3]);
      return 1;
    }

    /* Check if we are missing coordinates for beacons */
    if ((int)beaconCrdVec.size() != rnx.num_beacons()) {
      const auto tmp_vec = rnx.beacons_4charids();
      for (auto it = tmp_vec.cbegin(); it != tmp_vec.cend(); ++it) {
        auto match =
            std::find_if(beaconCrdVec.begin(), beaconCrdVec.end(),
                         [it](const dso::Sinex::SiteCoordinateResults &bcrd) {
                           return !std::strcmp(*it, bcrd.msite.site_code());
                         });
        if (match == beaconCrdVec.end()) {
          fprintf(stderr, "failed extracting coordinates for site %s\n", *it);
        }
      }
    }

    /* add PCO eccentricities to beacon coordinates; from now on, we will be
    referencing the L iono-free phase center
    */
    for (auto it = beaconCrdVec.begin(); it != beaconCrdVec.cend(); ++it) {
      Eigen::Vector3d pco_2GHz_enu, pco_400MHz_enu;
      dso::antenna_pco_enu(
          dso::char2doris_ground_antenna_type(it->msite.site_code()[3]),
          pco_2GHz_enu, pco_400MHz_enu);
      const auto if_ecc =
          pco_2GHz_enu + (pco_2GHz_enu - pco_400MHz_enu) /
                             (dso::SatelliteSystemTraits<
                                  dso::SATELLITE_SYSTEM::DORIS>::GAMMA_FACTOR -
                              1e0);
      /* rotation matrix R: (dX,dY,dZ) = R * (e,n,u) */
      const Eigen::Matrix3d R =
          dso::lvlh(dso::CartesianCrdConstView(it->cartesian_crd()));
      /* add PCO */
      const auto rsta = it->cartesian_crd().mv + R * if_ecc;
      it->x = rsta(0);
      it->y = rsta(1);
      it->z = rsta(2);
    }

    /* Tropospheric Handling -> VMF3
     * -----------------------------------------------------------------------
     */
    dso::Vmf3SiteHandler vmf3(argv[5]);
    for (const auto &beacon : beaconCrdVec) {
      if (vmf3.append_site(beacon.msite.site_code(), beacon.cartesian_crd())) {
        fprintf(stderr, "ERROR: Failed adding site %s to list of VMF3 sites!\n",
                beacon.msite.site_code());
        return 1;
      }
    }

    /* Construct integrator from YAML */
    auto t2 = t_start.add_seconds(dso::seconds(5 * 86400));
    dso::IntegrationParameters params = dso::IntegrationParameters::from_config(
        argv[1], dso::MjdEpoch(t_start), dso::MjdEpoch(t2));

    /* Initialize the integrator */
    dso::Dop853 dop853(satacc, 6 + NSTM + NSENS, &params, 1e-9, 1e-12);
    dop853.set_stiffness_check(10);

    /* Setup the Kalman filter */
    Eigen::Matrix<double, 6, 1> sat_inertial_state;
    Eigen::Matrix<double, 6, 1> sat_terrestrial_state;
    Kalman filter(6 + 2 + (int)beaconCrdVec.size(), dso::MjdEpoch(t_start));
    {
      /* Initial state (r, v) from Sp3 file at t0 (sometime before t_start ...)
       */
      auto t0 = t_start;
      {
        dso::Sp3c sp3(argv[6]);
        dso::Sp3Iterator sp3_iterator(sp3);
        if (sp3_iterator.goto_epoch(t_start, &t0)) {
          fprintf(stderr, "ERROR Failed to get reference position from SP3\n");
          return 1;
        }
        Eigen::VectorXd itrf = Eigen::Matrix<double, 6, 1>::Zero();
        sat_terrestrial_state(0) = sp3_iterator.data_block().state[0] * 1e3;
        sat_terrestrial_state(1) = sp3_iterator.data_block().state[1] * 1e3;
        sat_terrestrial_state(2) = sp3_iterator.data_block().state[2] * 1e3;
        sat_terrestrial_state(3) = sp3_iterator.data_block().state[4] * 1e-1;
        sat_terrestrial_state(4) = sp3_iterator.data_block().state[5] * 1e-1;
        sat_terrestrial_state(5) = sp3_iterator.data_block().state[6] * 1e-1;
        sat_inertial_state =
            itrf2gcrf(dso::MjdEpoch(t0), sat_terrestrial_state, &params);
      }

      filter.x_.segment<6>(0) = sat_inertial_state;
      for (int i = 0; i < (int)beaconCrdVec.size(); ++i)
        filter.reset_rfo(i, 0.0, 1e-11);
      /* scaling factors initial values */
      filter.reset_Cd(1.0, 0.2);
      filter.reset_Cr(1.0, 0.2);
      filter.tai() = dso::MjdEpoch(t0);
    }

    /* we will be using L1 & L2 from the RINEX file; get their indexes */
    const auto idx2Ghz = rnx.observation_index(
        dso::DorisObservationCode(DorisObservationType::phase, 1));
    const auto idx400Mhz = rnx.observation_index(
        dso::DorisObservationCode(DorisObservationType::phase, 2));
    const auto idxF = rnx.observation_index(
        dso::DorisObservationCode(DorisObservationType::frequency_offset, 0));
    assert((idx2Ghz >= 0 && idx400Mhz >= 0) && (idxF >= 0));

    /* On-board eccentricities: we will need a way to translate between CoM of
    the satellite and the iono-free phase center. Here is the "static" part of
    this transformation, i.e. the vector from the iono-free phase center to
    the CoM (in the satellite frame). Before using it, we will need to rotate
    it to the inertial frame using the satellite's attitude.
    Eccentricity acts like:
    CoM = L_iono-free - Δecc in [m],
    where Δecc is what we compute here
    */
    Eigen::Vector3d Decc_bf = Eigen::Vector3d::Zero();
    {
      /* 2GHz eccentricity in satellite frame */
      const auto L2GHz_ecc =
          dso::payload_eccentricity_bf<dso::SATELLITE_SYSTEM::DORIS>(
              dso::translate_satid("ja3"), "S1");
      /* 400MHz eccentricity in satellite frame */
      const auto L400MHz_ecc =
          dso::payload_eccentricity_bf<dso::SATELLITE_SYSTEM::DORIS>(
              dso::translate_satid("ja3"), "U2");
      /* CoM eccentricity in satellite frame */
      const auto CoM_ecc =
          dso::payload_eccentricity_bf<dso::SATELLITE_SYSTEM::DORIS>(
              dso::translate_satid("ja3"), "CoM");
      /* L_iono-free eccentricity in satellite frame:
      * [2 GHz phase center] + [vector from the 2 GHz phase center to the
      iono-free phase center], Lemoine et al. 2016, Eq. (20)
      */
      const auto Lif_ecc =
          L2GHz_ecc + (L2GHz_ecc - L400MHz_ecc) /
                          (dso::SatelliteSystemTraits<
                               dso::SATELLITE_SYSTEM::DORIS>::GAMMA_FACTOR -
                           1e0);
      /* (on-board) CoM to iono-free phase center: */
      Decc_bf = Lif_ecc - CoM_ecc;
    }

    /* a vector to hold previous obs for every beacon */
    std::vector<BeaconObs> last_obs;
    last_obs.reserve(rnx.num_beacons());

    char dbf[64];
    /* Iterate RINEX observations */
    for (auto bit = rnx.begin(); bit != rnx.end(); ++bit) {
      /* TAI of observation block */
      const auto tai = bit->mheader.corrected_epoch();
      printf("Consuming new RINEX epoch @%s\n",
             dso::to_char<dso::YMDFormat::YYYYMMDD, dso::HMSFormat::HHMMSSF>(
                 tai, dbf));
      const dso::MjdEpoch ttai(tai);
      /* eccentricity to L(iono-free) phase center (from CoM) */
      Eigen::Vector3d Decc_inertial = Eigen::Vector3d::Zero();
      /* state-transition-matrix from previous (integrator) epoch to now */
      Eigen::Matrix<double, 6, 6> STM;
      {
        /* integrate EoM+STM to now */
        params.t0() = filter.tai();
        const auto dsec =
            ttai.diff<dso::DateTimeDifferenceType::FractionalSeconds>(
                    params.t0())
                .seconds();
        Eigen::VectorXd y0stm(6 + NSTM + NSENS), ystm(6 + NSTM + NSENS);
        y0stm.setZero();
        y0stm.segment<6>(0) = filter.state_vector_gcrf();
        y0stm.segment(6, NSTM) =
            Eigen::Matrix<double, 6, 6>::Identity().reshaped(NSTM, 1);
        // Φxs(t0)=0
        y0stm.segment(6 + NSTM, NSENS).setZero();

        printf("Integrating ...\n");
        printf("\tfrom ref. epoch %s to %.6f[sec] away\n",
               dso::to_char<dso::YMDFormat::YYYYMMDD, dso::HMSFormat::HHMMSSF>(
                   params.t0(), dbf),
               dsec);

        if (dop853.integrate(0e0, dsec, y0stm, ystm)) {
          fprintf(stderr, "ERROR. Integrator failed!\n");
          return -99;
        }
        printf("\tIC: r=(%+.3f %+.3f %+.3f), v=(%+.6f %.6f %.6f) TAI=%s\n",
               y0stm(0), y0stm(1), y0stm(2), y0stm(3), y0stm(4), y0stm(5),
               dso::to_char<dso::YMDFormat::YYYYMMDD, dso::HMSFormat::HHMMSSF>(
                   params.t0(), dbf));
        printf("\tSol:r=(%+.3f %+.3f %+.3f), v=(%+.6f %.6f %.6f) TAI=%s\n",
               ystm(0), ystm(1), ystm(2), ystm(3), ystm(4), ystm(5),
               dso::to_char<dso::YMDFormat::YYYYMMDD, dso::HMSFormat::HHMMSSF>(
                   params.t0().add_seconds(dso::FractionalSeconds(dsec)), dbf));
        sat_inertial_state = ystm.segment<6>(0);

        /* kalman filter time update step */
        assert(
            std::abs(
                params.t0()
                    .add_seconds(dso::FractionalSeconds(dsec))
                    .diff<dso::DateTimeDifferenceType::FractionalSeconds>(ttai)
                    .seconds()) < 1e-12);
        filter.time_update(ttai, ystm, q_drag, q_srp, STM);

        /*
         CoM = L_iono-free - Δecc in [m],
        */
        // if (params.matt) {
        //   Decc_inertial = params.matt->rotate_payload_eccentricity(
        //       ttai.tai2tt(), Decc_bf, nullptr);
        // }
        Decc_inertial = Decc_bf;
      }

      /* store indexes of elements in last_obs that we updated; why? because
        at the end of this block, once we have the final state estimates, we
        must revisit them and update with the final estimates (i.e. considering
        updates from observations to every beacon in the block) */
      std::vector<int> beacon_indexes_updated;

      /* iterate through each beacon (of current block) */
      for (const auto &bcn : bit->mbeacon_obs) {
        /* "updated" rsat in different reference points X(tai)- or X(tai)+ */
        const Eigen::Vector3d rsat_gcrf_com = filter.position();
        const Eigen::Vector3d rsat_gcrf_lif = rsat_gcrf_com + Decc_inertial;
        const Eigen::Vector3d rsat_itrf_lif =
            gcrf2itrf(ttai, rsat_gcrf_lif, &params);
        /* get a pointer to the current Beacon (within rnx's m_stations) */
        const auto bcn_ptr = rnx.find_beacon_by_code(bcn.m_beacon_id);
        /* get beacon coordinates (match vs beaconCrdVec from SINEX) */
        const auto match = std::find_if(
            beaconCrdVec.cbegin(), beaconCrdVec.cend(),
            [&rnx = std::as_const(rnx), &bcn_ptr = std::as_const(bcn_ptr)](
                const dso::Sinex::SiteCoordinateResults &bcrd) {
              return !std::strcmp(bcn_ptr->id(), bcrd.msite.site_code());
            });
        if (match == beaconCrdVec.cend()) {
          fprintf(stderr, "failed extracting coordinates for site %s\n",
                  rnx.id2name(bcn.m_beacon_id));
        } else {
          /* index of beacon in beaconCrdVec -- indexing for Kalman -- */
          const int bcn_idx = std::distance(beaconCrdVec.cbegin(), match);
          /* beacon coordinates in ITRF */
          const auto rsta_itrf = match->cartesian_crd().copy_vec3d();
          /* beacon coordinates in GCRF */
          const auto rsta_gcrf = itrf2gcrf(ttai, rsta_itrf, &params);
          /* rotation matrix R: (dX,dY,dZ) = R * (e,n,u) */
          const Eigen::Matrix3d R =
              dso::lvlh(dso::CartesianCrdConstView(rsta_itrf));
          /* satellite to beacon vector, topocentric */
          const Eigen::Vector3d r_enu =
              R.transpose() * (rsat_itrf_lif - rsta_itrf);
          /* azimouth and elevation */
          const double az = std::atan2(r_enu(0), r_enu(1));
          /* clamp against tiny roundoff excursions (e.g. 1.0000000002) cause if
           * the asin argument is not in [-1,1], it will return Nan */
          const double sarg = std::clamp(
              r_enu(2) / (rsat_itrf_lif - rsta_itrf).norm(), -1e0, 1e0);
          const double el = std::asin(sarg);
          printf("Consuming new observation @beacon %s: el=%.3f, Az=%.2f\n",
                 bcn_ptr->id(), dso::rad2deg(el), dso::rad2deg(az));
          assert(el >= -1e-10 && el <= dso::deg2rad(90.01));
          if (dso::rad2deg(el) > MIN_ELEVATION_DEG) {
            /* elevation > cut-off angle, procced with observation */
            const double L2GHz = bcn.fetchv(idx2Ghz);
            const double L400MHz = bcn.fetchv(idx400Mhz);
            /* nominal frequencies of the beacon */
            double f2GHz, f400MHz;
            DorisSystem::beacon_nominal_frequency(bcn_ptr->m_shift_factor,
                                                  f2GHz, f400MHz);
            /* Construct the Iono-Free linear comb. */
            const double gamma = dso::SatelliteSystemTraits<
                dso::SATELLITE_SYSTEM::DORIS>::GAMMA_FACTOR;
            const double sqrt_gamma = dso::SatelliteSystemTraits<
                dso::SATELLITE_SYSTEM::DORIS>::GAMMA_FACTOR_SQRT;
            const double Lif =
                (gamma * L2GHz - sqrt_gamma * L400MHz) / (gamma - 1e0);
            const double fen = f2GHz;
            const double frn = f2GHz;
            /* correct nominal receiver frequency -> true receiver frequency
             */
            const double deltaFr = bcn.fetchv(idxF);
            const double frt = frn * (1e0 + deltaFr * 1e-11);
            /* compute tropospheric effect/correction (for range) */
            dso::vmf3::Vmf3Result trpres;
            if (vmf3.vmf3(bcn_ptr->id(), dso::MjdEpoch(tai), el, trpres)) {
              fprintf(stderr,
                      "Error. Failed computing vmf3 for site %s at t=%.9f\n",
                      bcn_ptr->id(), dso::MjdEpoch(tai).as_mjd());
              return 9;
            }
            /* find entry for previous obs for this beacon */
            auto prev = std::find_if(
                last_obs.begin(), last_obs.end(), [=](const BeaconObs &bo) {
                  return std::addressof(*bcn_ptr) == bo.bcn_ptr;
                });
            /* case A: no entry in the last_obs vector */
            if (prev == last_obs.end()) {
              last_obs.push_back(BeaconObs{
                  std::addressof(*bcn_ptr), rsat_gcrf_lif, rsta_gcrf, ttai,
                  trpres, az, el, Lif, L2GHz, L400MHz, fen, frt, 0});
              beacon_indexes_updated.push_back(last_obs.size() - 1);
              printf("Beacon is new, adding obs to stack\n");
            } else {
              const auto dtau =
                  ttai.diff<dso::DateTimeDifferenceType::FractionalSeconds>(
                          prev->tai)
                      .seconds();
              /* case B: previous entry exists but its in a previous pass */
              if (dtau >= _MIN_DIFF_FOR_NEW_ARC_) {
                prev->replace_data(rsat_gcrf_lif, rsta_gcrf, ttai, az, el, Lif,
                                   L2GHz, L400MHz, fen, frt, trpres);
                prev->pass_nr += 1;
                beacon_indexes_updated.push_back(
                    std::distance(last_obs.begin(), prev));
                printf("Last beacon obs is %.3f [sec] away, adding obs to new "
                       "pass\n",
                       dtau);
              } else if (dtau >= _MIN_DIFF_FOR_DOPPLER_) {
                /* case C: this is not a new pass, but we have missed a couple
                 * of obs */
                prev->replace_data(rsat_gcrf_lif, rsta_gcrf, ttai, az, el, Lif,
                                   L2GHz, L400MHz, fen, frt, trpres);
                beacon_indexes_updated.push_back(
                    std::distance(last_obs.begin(), prev));
                printf("Last beacon obs is %.3f [sec] away, re-initializing "
                       "due to Loss-Of-Lock\n",
                       dtau);
              } else {
                /* case D: we are fine, procced to form obs. equation */
                BeaconObs nowobs{prev->bcn_ptr,
                                 rsat_gcrf_lif,
                                 rsta_gcrf,
                                 ttai,
                                 trpres,
                                 az,
                                 el,
                                 Lif,
                                 L2GHz,
                                 L400MHz,
                                 fen,
                                 frt,
                                 prev->pass_nr};
                const double dfe = filter.rfo(bcn_idx);
                /* sigma of observation */
                const double sigma = OBS_SIGMA_0 / std::sin(el);
                /* observed value (obs. equation) */
                const double observed = prev->vobserved(nowobs) * (+1.0);
                /* computed value (obs. equation) */
                Eigen::VectorXd H(7);
                const double computed = prev->vcomputed(nowobs, STM, dfe, H);
                /* debug print */
                printf("Observation@%s %s observed=%+.3f computed=%.3f "
                       "residual=%.3f\n",
                       dso::to_char<dso::YMDFormat::YYYYMMDD,
                                    dso::HMSFormat::HHMMSSF>(ttai, dbf),
                       bcn_ptr->id(), observed, computed, observed - computed);
                /* observation update */
                Eigen::VectorXd dH = Eigen::VectorXd(filter.num_params());
                dH.setZero();
                dH.segment<6>(0) = H.segment<6>(0);
                dH(filter._rfo_start_index + bcn_idx) = H(6);
                filter.observation_update(observed, computed, sigma, dH);
                prev->replace_data(rsat_gcrf_lif, rsta_gcrf, ttai, az, el, Lif,
                                   L2GHz, L400MHz, fen, frt, trpres);
                beacon_indexes_updated.push_back(
                    std::distance(last_obs.begin(), prev));
              }
            } // previous obs. exists for beacon (aka prev != last_obs.end())
          } // elevation >= MIN_ELEVATION
        } // if (match == beaconCrdVec.cend())
      } // for (const auto &bcn : bit->mbeacon_obs)

      /* reset satellite state in the updated elements of last_obs */
      for (auto i : beacon_indexes_updated) {
        last_obs[i].rsat_gcrf = filter.position() + Decc_inertial;
      }
      /* update integration parameters with best values for dynamic params */
      params.Cd() = filter.drag_scale();
      params.Cr() = filter.srp_scale();

      /* DEBUG print */
      //{
      //  const Eigen::Matrix<double,6,1> pstate = gcrf2itrf(ttai,
      //  filter.state_vector_gcrf(), &params); char buf[64]; printf("RINEX
      //  epoch ended@%s r=(%.3f %.3f %.3f) v=(%.6f %.6f %.6f)\n",
      //      dso::to_char<dso::YMDFormat::YYYYMMDD,
      //          dso::HMSFormat::HHMMSSF>(
      //          ttai, buf),
      //      pstate(0), pstate(1), pstate(2), pstate(3), pstate(4),
      //      pstate(5));
      //}

    } /* for (auto it = rnx.begin(); it != rnx.end(); ++it) */
  } catch (std::exception &e) {
    fprintf(stderr, "Exception caugh! what is [%s]\n", e.what());
    return 9;
  }

  return 0;

} /* end main */

Eigen::Matrix<double, 6, 1>
gcrf2itrf(const dso::MjdEpoch &tai,
          const Eigen::Matrix<double, 6, 1> &state_gcrf,
          dso::IntegrationParameters *params) noexcept {

  const auto tt = tai.tai2tt();

  /* GCRF/ITRF at t */
  double fargs[14];
  dso::EopRecord eopr;
  Eigen::Quaterniond q_c2tirs, q_tirs2i;
  prep_c2i(tt.tt2tai(), params->eops(), q_c2tirs, q_tirs2i, fargs, eopr);
  Eigen::Vector3d omega;
  omega << 0e0, 0e0, dso::earth_rotation_rate(eopr.lod());

  /* state in ITRF (from GCRF) */
  Eigen::Matrix<double, 6, 1> itrf = Eigen::Matrix<double, 6, 1>::Zero();
  itrf.segment<3>(0) = q_tirs2i * (q_c2tirs * state_gcrf.segment<3>(0));
  itrf.segment<3>(3) =
      q_tirs2i * (q_c2tirs * state_gcrf.segment<3>(3) -
                  omega.cross(q_c2tirs * state_gcrf.segment<3>(0)));

  return itrf;
}

Eigen::Matrix<double, 3, 1>
gcrf2itrf(const dso::MjdEpoch &tai,
          const Eigen::Matrix<double, 3, 1> &state_gcrf,
          dso::IntegrationParameters *params) noexcept {

  const auto tt = tai.tai2tt();

  /* GCRF/ITRF at t */
  double fargs[14];
  dso::EopRecord eopr;
  Eigen::Quaterniond q_c2tirs, q_tirs2i;
  prep_c2i(tt.tt2tai(), params->eops(), q_c2tirs, q_tirs2i, fargs, eopr);

  /* state in ITRF (from GCRF) */
  Eigen::Matrix<double, 3, 1> itrf = Eigen::Matrix<double, 3, 1>::Zero();
  itrf = q_tirs2i * (q_c2tirs * state_gcrf);

  return itrf;
}

Eigen::Matrix<double, 6, 1>
itrf2gcrf(const dso::MjdEpoch &tai,
          const Eigen::Matrix<double, 6, 1> &state_itrf,
          dso::IntegrationParameters *params) noexcept {
  const auto tt = tai.tai2tt();

  /* GCRF/ITRF at t */
  double fargs[14];
  dso::EopRecord eopr;
  Eigen::Quaterniond q_c2tirs, q_tirs2i;
  prep_c2i(tt.tt2tai(), params->eops(), q_c2tirs, q_tirs2i, fargs, eopr);
  Eigen::Vector3d omega;
  omega << 0e0, 0e0, dso::earth_rotation_rate(eopr.lod());

  /* state in GCRF (from ITRF) */
  Eigen::Matrix<double, 6, 1> gcrf = Eigen::Matrix<double, 6, 1>::Zero();
  gcrf.segment<3>(0) =
      q_c2tirs.conjugate() * (q_tirs2i.conjugate() * state_itrf.segment<3>(0));
  gcrf.segment<3>(3) =
      q_c2tirs.conjugate() *
      (q_tirs2i.conjugate() * state_itrf.segment<3>(3) +
       omega.cross(q_tirs2i.conjugate() * state_itrf.segment<3>(0)));

  return gcrf;
}

Eigen::Matrix<double, 3, 1>
itrf2gcrf(const dso::MjdEpoch &tai,
          const Eigen::Matrix<double, 3, 1> &state_itrf,
          dso::IntegrationParameters *params) noexcept {
  const auto tt = tai.tai2tt();

  /* GCRF/ITRF at t */
  double fargs[14];
  dso::EopRecord eopr;
  Eigen::Quaterniond q_c2tirs, q_tirs2i;
  prep_c2i(tt.tt2tai(), params->eops(), q_c2tirs, q_tirs2i, fargs, eopr);

  /* position in GCRF (from ITRF) */
  Eigen::Matrix<double, 3, 1> gcrf = Eigen::Matrix<double, 3, 1>::Zero();
  gcrf.segment<3>(0) =
      q_c2tirs.conjugate() * (q_tirs2i.conjugate() * state_itrf.segment<3>(0));
  return gcrf;
}

int satacc(double tsec, Eigen::Ref<const Eigen::VectorXd> y0,
           dso::IntegrationParameters *params,
           Eigen::Ref<Eigen::VectorXd> yp) noexcept {
  /* epoch of request in TT */
  const auto tt =
      (params->t0().add_seconds(dso::FractionalSeconds(tsec))).tai2tt();
  // printf("\tGCRF state: r=(%+.3f,%+.3f,%.3f) v=(%+.6f,%+.6f,%+.6f)\n", y0(0),
  // y0(1), y0(2), y0(3), y0(4), y0(5));

  /* GCRS/ITRS at t */
  double fargs[14];
  dso::EopRecord eopr;
  Eigen::Quaterniond q_c2tirs, q_tirs2i;
  prep_c2i(tt.tt2tai(), params->eops(), q_c2tirs, q_tirs2i, fargs, eopr);
  Eigen::Vector3d omega;
  omega << 0e0, 0e0, dso::earth_rotation_rate(eopr.lod());
  Eigen::Matrix3d R = (q_tirs2i * q_c2tirs).toRotationMatrix(); // GCRS -> ITRS

  /* get Sun position & velocity in ICRF */
  Eigen::Matrix<double, 6, 1> rsun;
  if (dso::planet_state(dso::Planet::SUN, tt, rsun)) {
    fprintf(stderr, "ERROR Failed to compute Sun position!\n");
    return 100;
  }

  /* get Moon position in ICRF */
  Eigen::Matrix<double, 3, 1> rmoon;
  if (dso::planet_pos(dso::Planet::MOON, tt, rmoon)) {
    fprintf(stderr, "ERROR Failed to compute Moon position!\n");
    return 101;
  }

  /* state in ITRS (from GCRS) */
  Eigen::Matrix<double, 6, 1> itrs = Eigen::Matrix<double, 6, 1>::Zero();
  itrs.segment<3>(0) = q_tirs2i * (q_c2tirs * y0.segment<3>(0));
  itrs.segment<3>(3) = q_tirs2i * (q_c2tirs * y0.segment<3>(3) -
                                   omega.cross(q_c2tirs * y0.segment<3>(0)));
  // printf("\tITRF state: r=(%+.3f,%+.3f,%.3f) v=(%+.6f,%+.6f,%+.6f)\n",
  // itrs(0), itrs(1), itrs(2), itrs(3), itrs(4), itrs(5));

  /* accumulated acceleration and gradient in ITRS */
  Eigen::Vector3d ai = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 3, 3> gi =
      Eigen::Matrix<double, 3, 3>::Zero(); /* -> aka da/dr [ITRS]*/
  /* accumulated acceleration in GCRS */
  Eigen::Vector3d ac = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 3, 3> gc =
      Eigen::Matrix<double, 3, 3>::Zero(); /* -> aka da/dr [GCRS]*/
  Eigen::Matrix<double, 3, 3> gv =
      Eigen::Matrix<double, 3, 3>::Zero(); /* -> aka da/dv [GCRS]*/

  /* accumulated SH coeffs
  TODO!! WARNING!! What if some other SH coeffs (e.g. ocean tide) have (n,m)>
  gravity(n,m)? write a function as member of IntegrationParameters that
  return a StokesCoeffs of some degree and order
  */
  auto acstokes{params->earth_gravity()};

  /* add Solid Earth Tides to SH coeffs */
  if (params->solid_earth_tide()) {
    params->solid_earth_tide()->stokes_coeffs(
        tt, tt.tt2ut1(eopr.dut()), q_tirs2i * (q_c2tirs * rmoon),
        q_tirs2i * (q_c2tirs * rsun.segment<3>(0)), fargs);
    /* add SET effect */
    acstokes += params->solid_earth_tide()->stokes_coeffs();
  }

  /* add Ocean Tides to SH coeffs */
  if (params->ocean_tide()) {
    params->ocean_tide()->stokes_coeffs(tt, tt.tt2ut1(eopr.dut()), fargs);
    acstokes += params->ocean_tide()->stokes_coeffs();
  }

  /* add Pole Tide to SH coeffs */
  if (params->pole_tide()) {
    double dC21, dS21;
    params->pole_tide()->stokes_coeffs(tt, eopr.xp(), eopr.yp(), dC21, dS21);
    acstokes.C(2, 1) += dC21;
    acstokes.S(2, 1) += dS21;
  }

  /* add Ocean Pole Tide to SH coeffs */
  if (params->ocean_pole_tide()) {
    if (params->ocean_pole_tide()->stokes_coeffs(tt, eopr.xp(), eopr.yp())) {
      fprintf(stderr, "ERROR Failed computing Stokes Coefficients\n");
      return 102;
    }
    acstokes += params->ocean_pole_tide()->stokes_coeffs();
  }

  /* add deAliasing to SH coeffs */
  if (params->dealias()) {
    /*
    TODO!! WARNING!! The dealias instance should have a function that appends
    the coefficients at a StokesCoeffs instance!
    */
    auto tempstokes{params->earth_gravity()};
    if (params->dealias()->coefficients_at(
            dso::from_mjdepoch<dso::nanoseconds>(tt), tempstokes)) {
      fprintf(stderr, "Failed interpolating dealiasing coefficients\n");
      return 103;
    }
    acstokes += tempstokes;
  }

  /* add atmospheric tides to SH coeffs */
  if (params->atmospheric_tide()) {
    params->atmospheric_tide()->stokes_coeffs(tt, tt.tt2ut1(eopr.dut()), fargs);
    acstokes += params->atmospheric_tide()->stokes_coeffs();
  }

  /* acceleration from accumulated SH expansion */
  if (dso::sh2gradient_cunningham(acstokes, itrs.segment<3>(0), ai, gi,
                                  params->earth_gravity().max_degree(),
                                  params->earth_gravity().max_order(), -1, -1,
                                  &(params->tw()), &(params->tm()))) {
    fprintf(stderr, "ERROR Failed computing acceleration/gradient\n");
    return 104;
  }
  // printf("\tGravity acceleration: %+.6f %+.6f %+.6f [ITRF, m/sec^2]\n",
  // ai(0), ai(1), ai(2));

  /* Third Body perturbations and Relativity (IERS 2010) */
  {
    ac += dso::point_mass_acceleration(y0.segment<3>(0), rsun.segment<3>(0),
                                       GM_Sun, gc);
    Eigen::Matrix<double, 3, 3> tmp;
    ac += dso::point_mass_acceleration(y0.segment<3>(0), rmoon, GM_Moon, tmp);
    /* add gradients (wrt r, aka da/dr -> 3x3) */
    gc += tmp;
    /* Relativistic Correction */
    ac += dso::iers2010_relativistic_acceleration(y0.segment<6>(0), rsun);
  }
  // printf("\t3d body acceleration: %+.6f %+.6f %+.6f [GCRF, m/sec^2]\n",
  // ac(0), ac(1), ac(2));

  /* setup vectors that (maybe) needed for attitude */
  Eigen::Vector3d avecs[3];
  avecs[0] = rsun.segment<3>(0) - y0.segment<3>(0);
  avecs[1] = y0.segment<3>(0);
  avecs[2] = y0.segment<3>(3);

  /* Atmospheric Drag */
  Eigen::Vector3d dadCd =
      Eigen::Vector3d::Zero(); /* unscaled drag acceleration */
  if (params->Cd()) {
    /* compute atmospheric density */
    const double density = params->matmdens->density(itrs.segment<3>(0), tt);
    /* density gradient aka dρ/dr */
    Eigen::Vector3d dpdr;
    {
      const double epsilon = 15e0; // in [m]
      const auto pp = itrs.segment<3>(0);
      /* density gradient */
      const double dp_epx = params->matmdens->density(
          pp + Eigen::Vector3d{epsilon, 0e0, 0e0}, tt);
      const double dp_emx = params->matmdens->density(
          pp + Eigen::Vector3d{-epsilon, 0e0, 0e0}, tt);
      const double dp_epy = params->matmdens->density(
          pp + Eigen::Vector3d{0e0, epsilon, 0e0}, tt);
      const double dp_emy = params->matmdens->density(
          pp + Eigen::Vector3d{0e0, -epsilon, 0e0}, tt);
      const double dp_epz = params->matmdens->density(
          pp + Eigen::Vector3d{0e0, 0e0, epsilon}, tt);
      const double dp_emz = params->matmdens->density(
          pp + Eigen::Vector3d{0e0, 0e0, -epsilon}, tt);
      dpdr << dp_epx - dp_emx, dp_epy - dp_emy, dp_epz - dp_emz;
      /* note: frame change -> grad(ρ)_{gcrs} = R^(T) * grad(ρ)_{itrs} */
      dpdr = R.transpose() * dpdr / (2e0 * epsilon);
    }
    /* compute satellite velocity w.r.t atmosphere */
    const Eigen::Vector3d omega_gcrs = q_c2tirs.conjugate() * omega;
    const auto vr = y0.segment<3>(3) - omega_gcrs.cross(y0.segment<3>(0));
    /* compute atmospheric drag acceleration and gradients (unscaled) */
    Eigen::Matrix<double, 3, 3> dadr, dadv;
    dadCd = dso::atmospheric_drag(params->matt->rotated_macromodel(tt, avecs),
                                  vr, density, params->matt->satellite_mass(),
                                  omega_gcrs, dpdr, dadr, dadv);
    /* add (scaled) acceleration to commulative */
    ac += params->Cd() * dadCd;
    /* commulative da/dr */
    gc += params->Cd() * dadr;
    /* note that gv is da/dv */
    gv += params->Cd() * dadv;
    // printf("\tdrag    acceleration: %+.6f %+.6f %+.6f [scaled by %.3f, GCRF,
    // m/sec^2]\n", dadCd(0), dadCd(1), dadCd(2), params->Cd());
  }

  /* Solar Radiation Pressure */
  Eigen::Vector3d dadCr = Eigen::Vector3d::Zero();
  if (params->Cr()) {
    /* occultation factor */
    const double of =
        dso::conical_occultation(y0.segment<3>(0), rsun.segment<3>(0));

    if (of > 0e0) {
      Eigen::Matrix<double, 3, 3> dadr;
      /* Solar Radiation Pressure (acceleration) */
      Eigen::Vector3d a0 = dso::solar_radiation_pressure(
          params->matt->rotated_macromodel(tt, avecs), y0.segment<3>(0),
          rsun.segment<3>(0), params->matt->satellite_mass(), dadr);
      dadCr = of * a0; // ∂a/∂Cr (Cr is the scale)
      ac += params->Cr() * dadCr;
      gc += params->Cr() * of * dadr;
      // printf("\tsrp     acceleration: %+.6f %+.6f %+.6f [scaled by %.3f*%.3f,
      // GCRF, m/sec^2]\n", dadCr(0), dadCr(1), dadCr(2), params->Cr(), of);
    }
  }

  /* form the derivative vector */
  yp.segment<3>(0) = y0.segment<3>(3);
  yp.segment<3>(3) = ac + q_c2tirs.conjugate() * (q_tirs2i.conjugate() * ai);

  /* handle STM */
  Eigen::Matrix<double, 6, 6> F0;
  F0.block<6, 1>(0, 0) = y0.segment<6>(6);
  F0.block<6, 1>(0, 1) = y0.segment<6>(12);
  F0.block<6, 1>(0, 2) = y0.segment<6>(18);
  F0.block<6, 1>(0, 3) = y0.segment<6>(24);
  F0.block<6, 1>(0, 4) = y0.segment<6>(30);
  F0.block<6, 1>(0, 5) = y0.segment<6>(36);

  Eigen::Matrix<double, 6, 6> dFdt;
  dFdt.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Zero();
  dFdt.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity();
  /* note: da/dr[gcrs] = R^T * da/dr[itrs] * R */
  dFdt.block<3, 3>(3, 0) = R.transpose() * gi * R + gc;
  dFdt.block<3, 3>(3, 3) = gv;

  // Extract Φxs(t) from y0
  Eigen::Matrix<double, 6, 2> S0;
  S0.col(0) = y0.segment<6>(6 + NSTM + 0 * 6);
  S0.col(1) = y0.segment<6>(6 + NSTM + 1 * 6);

  // Build A = dFdt (your existing 6x6)
  Eigen::Matrix<double, 6, 6> A = dFdt;

  // Build B (6x2)
  Eigen::Matrix<double, 6, 2> B;
  B.setZero();
  B.block<3, 1>(3, 0) = dadCd; // ∂v̇/∂sD
  B.block<3, 1>(3, 1) = dadCr; // ∂v̇/∂sS

  // Sensitivity ODE: dS/dt = A*S + B
  Eigen::Matrix<double, 6, 2> dSdt = A * S0 + B;

  // Pack into yp after STM part
  yp.segment<6>(6 + 36 + 0 * 6) = dSdt.col(0);
  yp.segment<6>(6 + 36 + 1 * 6) = dSdt.col(1);

  const auto F = dFdt * F0;
  for (int i = 0; i < 6; i++) {
    yp.segment<6>(6 + i * 6) = F.block<6, 1>(0, i);
  }

  return 0;
}

/* Compute relevant quaternions for the ITRS/GCRS transformation */
int prep_c2i(const dso::MjdEpoch &tai, dso::EopSeries &eops,
             Eigen::Quaterniond &q_c2tirs, Eigen::Quaterniond &q_tirs2i,
             double *fargs, dso::EopRecord &eopr) noexcept {

  /* epoch of request in TT */
  const auto tt = tai.tai2tt();

  /* compute (X,Y) CIP and fundamental arguments (we are doing this here
   * to compute fargs).
   */
  double Xcip, Ycip;
  dso::xycip06a(tt, Xcip, Ycip, fargs);

  /* interpolate EOPs */
  if (dso::EopSeries::out_of_bounds(eops.interpolate(tt, eopr))) {
    fprintf(stderr, "Failed to interpolate: Epoch is out of bounds!\n");
    return 1;
  }

  /* compute gmst using an approximate value for UT1 (linear interpolation) */
  double dut1_approx;
  eops.approx_dut1(tt, dut1_approx);
  const double gmst = dso::gmst(tt, tt.tt2ut1(dut1_approx));

  /* add libration effect [micro as] */
  {
    double dxp, dyp, dut1, dlod;
    dso::deop_libration(fargs, gmst, dxp, dyp, dut1, dlod);
    eopr.xp() += dxp * 1e-6;
    eopr.yp() += dyp * 1e-6;
    eopr.dut() += dut1 * 1e-6;
    eopr.lod() += dlod * 1e-6;
  }

  /* add ocean tidal effect [micro as] */
  {
    double dxp, dyp, dut1, dlod;
    dso::deop_ocean_tide(fargs, gmst, dxp, dyp, dut1, dlod);
    eopr.xp() += dxp * 1e-6;
    eopr.yp() += dyp * 1e-6;
    eopr.dut() += dut1 * 1e-6;
    eopr.lod() += dlod * 1e-6;
  }

  /* de-regularize */
  {
    double ut1_cor;
    double lod_cor;
    double omega_cor;
    dso::deop_zonal_tide(fargs, ut1_cor, lod_cor, omega_cor);
    /* apply (note: microseconds to seconds) */
    eopr.dut() += (ut1_cor * 1e-6);
    eopr.lod() += (lod_cor * 1e-6);
  }

  /* use fundamental arguments to compute s */
  const double s = dso::s06(tt, Xcip, Ycip, fargs);

  /* apply CIP corrections */
  Xcip += dso::sec2rad(eopr.dX());
  Ycip += dso::sec2rad(eopr.dY());

  /* spherical crd for CIP (E, d) */
  double d, e;
  dso::detail::xycip2spherical(Xcip, Ycip, d, e);

  /* Earth rotation angle */
  const double era = dso::era00(tt.tt2ut1(eopr.dut()));

  /* compute rotation quaternions */
  q_c2tirs = dso::detail::c2tirs(era, s, d, e);
  q_tirs2i = dso::detail::tirs2i(dso::sec2rad(eopr.xp()),
                                 dso::sec2rad(eopr.yp()), dso::sp00(tt));

  return 0;
}
