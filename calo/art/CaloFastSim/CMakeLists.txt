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

cet_build_plugin(CaloResponseDumper art::module
    REG_SOURCE src/CaloResponseDumper_module.cc
    LIBRARIES REG
      Offline::MCDataProducts
      Offline::CalorimeterGeom
      Offline::GeometryService
      Offline::DataProducts
      Offline::CaloConditions
      Offline::DAQConditions
      Offline::GlobalConstantsService
      Offline::ProditionsService
      Offline::DbTables
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

cet_build_plugin(CaloPropagationDumper art::module
    REG_SOURCE src/CaloPropagationDumper_module.cc
    LIBRARIES REG
      Offline::CaloMC
      Offline::ConfigTools
      Offline::CalorimeterGeom
      Offline::GeometryService
      art::Framework_Core
      art::Framework_Principal
      art::Framework_Services_Registry
      art::Utilities
      art_root_io::TFileService_service
      canvas::canvas
      fhiclcpp::fhiclcpp
      fhiclcpp::types
      cetlib::cetlib
      cetlib_except::cetlib_except
      CLHEP::CLHEP
      ROOT::Tree
      ROOT::Hist
      ROOT::RIO
)

install_source(SUBDIRS src)
install_fhicl(SUBDIRS fcl SUBDIRNAME CaloFastSim/fcl)
