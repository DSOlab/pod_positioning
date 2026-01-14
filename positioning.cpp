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
#include "sp3/sv_interpolate.hpp"

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
constexpr const double MIN_ELEVATION_DEG = 5.;

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
  Eigen::Vector3d rsat_itrf;
  Eigen::Vector3d rsta_itrf;
  dso::MjdEpoch tai;
  dso::vmf3::Vmf3Result vmf3;
  double az, el;
  double Lif, L2GHz, L400MHz;
  double fen, frt;
  int pass_nr{0};

  double
  dTropo(double zwd = std::numeric_limits<double>::min()) const noexcept {
    zwd = (zwd == std::numeric_limits<double>::min()) ? vmf3.zwd() : zwd;
    return vmf3.zhd() * vmf3.mfh() + zwd * vmf3.mfw();
  }

  void replace_data(const Eigen::Vector3d &_rsat_itrf,
                    const Eigen::Vector3d &_rsta_itrf,
                    const dso::MjdEpoch &_tai, double _az, double _el,
                    double _Lif, double _L2GHz, double _L400MHz, double _fen,
                    double _frt, const dso::vmf3::Vmf3Result &_vmf3) noexcept {
    this->rsat_itrf = _rsat_itrf;
    this->rsta_itrf = _rsta_itrf;
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
    // TODO what is the sign for the tropospheric bias ?
    const double Vm = -(C / fen) * (fen - mfrt - Ndop / dtau) +
                      (next.dTropo() - this->dTropo()) / dtau; // [m/sec]
    return -Vm;                                                // [m/sec]
  }

  /* Equation (13) from Lemoine et al, 2010
   * Note that the calling instance is at the "point of linearization"
   */
  double vcomputed(const BeaconObs &next, double dfe,
                   Eigen::VectorXd &H) const noexcept {
    constexpr const double C = iers2010::C;
    const double dtau =
        next.tai.diff<dso::DateTimeDifferenceType::FractionalSeconds>(this->tai)
            .seconds();                       // [sec]
    const double Ndop = next.Lif - this->Lif; // [cycles]]
    const double mfrt = (this->frt + next.frt) / 2.;

    const Eigen::Vector3d u1 = this->rsat_itrf - next.rsta_itrf;
    const double rho1 = u1.norm();

    const Eigen::Vector3d u2 = next.rsat_itrf - next.rsta_itrf;
    const double rho2 = u2.norm();

    const double Vc =
        (rho2 - rho1) / dtau - (C / fen) * (Ndop / dtau + mfrt) * dfe;

    /* partials matrix, H = [dh/dr, dh/ddfe] */
    Eigen::VectorXd b1 = u1 / rho1;
    Eigen::VectorXd b2 = u2 / rho2;
    H.setZero(4);
    H.segment<3>(0) = -(b2 - b1) / dtau;
    H(3) = -(C / fen) * (Ndop / dtau + mfrt);

    return Vc; // [m/sec]
  }

}; /* BeaconObs */

struct Kalman {
  /* number of estimated parameters:
   * 3 * num of beacons -> cartesian coordinates of beacon (X,Y,Z) in [m]
   * 1 * num of beacons -> (running) relative frequency offset per beacon
   *                       starting at index #8
   */
  int num_beacons{0};
  dso::MjdEpoch tai_;
  Eigen::VectorXd x_; /* estimates */
  Eigen::MatrixXd P_; /* P matrix */

  int num_params() const noexcept { return x_.rows(); }

  Kalman(int numBeacons, const dso::MjdEpoch &_tai = dso::MjdEpoch{}) noexcept
      : num_beacons(numBeacons), tai_(_tai),
        x_(Eigen::VectorXd::Zero(numBeacons * 4)),
        P_(Eigen::MatrixXd::Identity(numBeacons * 4, numBeacons * 4)) {}

  int _rfo_start_index() const noexcept { return num_beacons * 3; }

  auto tai() const noexcept { return tai_; }
  auto &tai() noexcept { return tai_; }

  const double &rfo(int beacon_index) const noexcept {
    return x_(_rfo_start_index() + beacon_index);
  }

  double &rfo(int beacon_index) noexcept {
    return x_(_rfo_start_index() + beacon_index);
  }

  int index_of_rfo(int beacon_index) const noexcept {
    return _rfo_start_index() + beacon_index;
  }
  int index_of_pos(int beacon_index) const noexcept { return beacon_index * 3; }

  Eigen::Vector3d beacon_pos(int beacon_index) const {
    return x_.segment<3>(beacon_index * 3);
  }

  void reset_rfo(int beacon_index, double val0, double sigma0) noexcept {
    const int i = beacon_index + _rfo_start_index();
    rfo(beacon_index) = val0;
    P_.row(i).setZero();
    P_.col(i).setZero();
    P_(i, i) = sigma0 * sigma0;
  }

  void reset_pos(int beacon_index, const Eigen::Vector3d &xyz,
                 double sigma0) noexcept {
    x_.segment<3>(beacon_index * 3) = xyz;
    // zero 3 rows/cols and set 3 variances
    const int i0 = beacon_index * 3;
    for (int k = 0; k < 3; k++) {
      P_.row(i0 + k).setZero();
      P_.col(i0 + k).setZero();
      P_(i0 + k, i0 + k) = sigma0 * sigma0;
    }
  }

  void time_update(const dso::MjdEpoch &tai) noexcept { tai_ = tai; }

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

    /* get integration parameters from YAML */
    auto t2 = t_start.add_seconds(dso::seconds(5 * 86400));
    dso::IntegrationParameters params = dso::IntegrationParameters::from_config(
        argv[1], dso::MjdEpoch(t_start), dso::MjdEpoch(t2));

    /* Initial state (r, v) from Sp3 file */
    Eigen::Matrix<double, 6, 1> sat_terrestrial_state, sat_inertial_state;
    dso::Sp3c sp3(argv[6]);
    dso::sp3::SatelliteId sv("L27");
    sv.set_id(sp3.sattellite_vector()[0].id);
    dso::SvInterpolator sv_intrp(sv, sp3);
    // if (sp3_iterator.goto_epoch(t_start.tai2tt())) {
    //   fprintf(stderr, "ERROR Failed to get reference position from SP3\n");
    //   return 1;
    // }
    // Eigen::VectorXd itrf = Eigen::Matrix<double, 6, 1>::Zero();
    // sat_terrestrial_state(0) = sp3_iterator.data_block().state[0] * 1e3;
    // sat_terrestrial_state(1) = sp3_iterator.data_block().state[1] * 1e3;
    // sat_terrestrial_state(2) = sp3_iterator.data_block().state[2] * 1e3;
    // sat_terrestrial_state(3) = sp3_iterator.data_block().state[4] * 1e-1;
    // sat_terrestrial_state(4) = sp3_iterator.data_block().state[5] * 1e-1;
    // sat_terrestrial_state(5) = sp3_iterator.data_block().state[6] * 1e-1;
    // sat_inertial_state =
    //     itrf2gcrf(dso::MjdEpoch(t_start), sat_terrestrial_state, &params);

    /* Setup the Kalman filter */
    Kalman filter((int)beaconCrdVec.size(), dso::MjdEpoch(t_start));
    {
      int idx = 0;
      for (const auto &bcr : beaconCrdVec) {
        filter.reset_pos(idx, bcr.cartesian_crd().copy_vec3d(), 5e0);
        ++idx;
      }
      for (int i = 0; i < (int)beaconCrdVec.size(); ++i)
        filter.reset_rfo(i, 0.0, 1e-11);
      /* scaling factors initial values */
      filter.tai() = dso::MjdEpoch(t_start);
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
      // Eigen::Vector3d Decc_inertial = Eigen::Vector3d::Zero();
      /* interpolate satellite coordinates/velocity */
      {
        double xyz[3] = {0}, dxdydz[3];
        if (sv_intrp.interpolate_at(tai, xyz, dxdydz)) {
          fprintf(stderr, "Interpolator failed!\n");
          return 5;
        }

        Eigen::VectorXd itrf = Eigen::Matrix<double, 6, 1>::Zero();
        sat_terrestrial_state(0) = xyz[0] * 1e3;
        sat_terrestrial_state(1) = xyz[1] * 1e3;
        sat_terrestrial_state(2) = xyz[2] * 1e3;
        sat_terrestrial_state(3) = dxdydz[0] * 1e-1;
        sat_terrestrial_state(4) = dxdydz[1] * 1e-1;
        sat_terrestrial_state(5) = dxdydz[2] * 1e-1;
        sat_inertial_state =
            itrf2gcrf(dso::MjdEpoch(tai), sat_terrestrial_state, &params);
        filter.time_update(ttai);
      }

      /* store indexes of elements in last_obs that we updated; why? because
        at the end of this block, once we have the final state estimates, we
        must revisit them and update with the final estimates (i.e. considering
        updates from observations to every beacon in the block) */
      std::vector<int> beacon_indexes_updated;

      /* iterate through each beacon (of current block) */
      for (const auto &bcn : bit->mbeacon_obs) {
        /* "updated" rsat in different reference points X(tai)- or X(tai)+ */
        // const Eigen::Vector3d rsat_gcrf_com = sat_inertial_state;
        // const Eigen::Vector3d rsat_gcrf_lif = rsat_gcrf_com;
        const Eigen::Vector3d rsat_itrf_lif =
            sat_terrestrial_state.segment<3>(0);
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
          // const auto rsta_itrf = match->cartesian_crd().copy_vec3d();
          const auto rsta_itrf = filter.beacon_pos(bcn_idx);
          /* beacon coordinates in GCRF */
          // const auto rsta_gcrf = itrf2gcrf(ttai, rsta_itrf, &params);
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
            printf(
                "Note Topospheric delay for %s@%s => ZHD=%.3f mhd=%.3f "
                "ZWD=%.3f mwd=%.3f\n",
                bcn_ptr->id(),
                dso::to_char<dso::YMDFormat::YYYYMMDD, dso::HMSFormat::HHMMSSF>(
                    ttai, dbf),
                trpres.zhd(), trpres.mfh(), trpres.zwd(), trpres.mfw());
            /* find entry for previous obs for this beacon */
            auto prev = std::find_if(
                last_obs.begin(), last_obs.end(), [=](const BeaconObs &bo) {
                  return std::addressof(*bcn_ptr) == bo.bcn_ptr;
                });
            /* case A: no entry in the last_obs vector */
            if (prev == last_obs.end()) {
              last_obs.push_back(BeaconObs{
                  std::addressof(*bcn_ptr), rsat_itrf_lif, rsta_itrf, ttai,
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
                prev->replace_data(rsat_itrf_lif, rsta_itrf, ttai, az, el, Lif,
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
                prev->replace_data(rsat_itrf_lif, rsta_itrf, ttai, az, el, Lif,
                                   L2GHz, L400MHz, fen, frt, trpres);
                beacon_indexes_updated.push_back(
                    std::distance(last_obs.begin(), prev));
                printf("Last beacon obs is %.3f [sec] away, re-initializing "
                       "due to Loss-Of-Lock\n",
                       dtau);
              } else {
                /* case D: we are fine, procced to form obs. equation */
                BeaconObs nowobs{prev->bcn_ptr,
                                 rsat_itrf_lif,
                                 rsta_itrf,
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
                /* sigma of observation */
                const double sigma = .5 / std::sin(el);
                /* observed value (obs. equation) */
                const double observed = prev->vobserved(nowobs) * (+1.0);
                /* computed value (obs. equation) */
                Eigen::VectorXd H(4);
                const double computed =
                    prev->vcomputed(nowobs, filter.rfo(bcn_idx), H);
                /* debug print */
                printf("Observation@%s %s observed=%+.3f computed=%.3f "
                       "residual=%.3f\n",
                       dso::to_char<dso::YMDFormat::YYYYMMDD,
                                    dso::HMSFormat::HHMMSSF>(ttai, dbf),
                       bcn_ptr->id(), observed, computed, observed - computed);
                /* observation update */
                Eigen::VectorXd dH = Eigen::VectorXd(filter.num_params());
                dH.setZero();
                dH.segment<3>(filter.index_of_pos(bcn_idx)) = H.segment<3>(0);
                dH(filter.index_of_rfo(bcn_idx)) = H(3);
                filter.observation_update(observed, computed, sigma, dH);
                prev->replace_data(rsat_itrf_lif, rsta_itrf, ttai, az, el, Lif,
                                   L2GHz, L400MHz, fen, frt, trpres);
                beacon_indexes_updated.push_back(
                    std::distance(last_obs.begin(), prev));
              }
            } // previous obs. exists for beacon (aka prev != last_obs.end())
          } // elevation >= MIN_ELEVATION
        } // if (match == beaconCrdVec.cend())
      } // for (const auto &bcn : bit->mbeacon_obs)

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
