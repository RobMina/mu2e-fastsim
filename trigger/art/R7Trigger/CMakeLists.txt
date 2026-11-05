cet_build_plugin(KalSeedSelectionFeatures art::module
    REG_SOURCE src/KalSeedSelectionFeatures_module.cc
    LIBRARIES REG
      Offline::RecoDataProducts
      Offline::DataProducts
      Offline::GeneralUtilities
      KinKal::Trajectory
      KinKal::General
      art::Framework_Core
      art::Framework_Principal
      art::Framework_Services_Registry
      art::Utilities
      art_root_io::TFileService_service
      art_root_io::tfile_support
      canvas::canvas
      fhiclcpp::fhiclcpp
      fhiclcpp::types
      cetlib::cetlib
      cetlib_except::cetlib_except
      ROOT::Core
      ROOT::RIO
      ROOT::Tree
      ROOT::MathCore
      ROOT::GenVector
      ROOT::Matrix
)

install_source(SUBDIRS src)
