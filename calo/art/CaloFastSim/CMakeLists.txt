cet_build_plugin(CaloLibraryDumper art::module
    REG_SOURCE src/CaloLibraryDumper_module.cc
    LIBRARIES REG
      Offline::MCDataProducts
      Offline::CalorimeterGeom
      Offline::GeometryService
      Offline::DataProducts
      art::Framework_Core
      art::Framework_Principal
      art::Framework_Services_Registry
      art::Utilities
      art_root_io::TFileService_service
      canvas::canvas
      messagefacility::MF_MessageLogger
      fhiclcpp::fhiclcpp
      fhiclcpp::types
      cetlib::cetlib
      cetlib_except::cetlib_except
      CLHEP::CLHEP
      ROOT::Tree
)

install_source(SUBDIRS src)
install_fhicl(SUBDIRS fcl SUBDIRNAME CaloFastSim/fcl)
