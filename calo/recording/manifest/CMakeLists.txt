# REVIEW SOURCE ONLY. Not connected to any parent build.
foreach(plugin CaloInputEventManifest CaloIncidentManifest)
  cet_build_plugin(${plugin} art::module
    REG_SOURCE src/${plugin}_module.cc
    LIBRARIES REG
      Offline::MCDataProducts
      art::Framework_Core
      art::Framework_Principal
      art::Framework_Services_Registry
      art::Utilities
      art_root_io::tfile_support
      art_root_io::TFileService_service
      canvas::canvas
      fhiclcpp::fhiclcpp
      cetlib::cetlib
      cetlib_except::cetlib_except
      CLHEP::CLHEP
      ROOT::Tree)
endforeach()
install_source(SUBDIRS src)
install_fhicl(SUBDIRS fcl SUBDIRNAME CaloIncidentManifest/fcl)
