cmake_minimum_required(VERSION 3.18...4.1)
project(m2fs_celer_adapter VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Celeritas 0.6.3 EXACT REQUIRED)
find_package(Geant4 11.3.2 EXACT REQUIRED)
if(NOT CELERITAS_USE_CUDA OR NOT CELERITAS_USE_VecGeom OR NOT CELERITAS_USE_covfie)
  message(FATAL_ERROR "Adapter requires the CUDA/VecGeom/covfie image")
endif()
set(APP_SOURCES
  app/celer-g4.cc app/ActionInitialization.cc app/DetectorConstruction.cc
  app/EventAction.cc app/GeantDiagnostics.cc app/GlobalSetup.cc
  app/HepMC3PrimaryGeneratorAction.cc app/LogHandlers.cc
  app/PGPrimaryGeneratorAction.cc app/RunAction.cc app/RunInput.cc
  app/RunInputIO.json.cc app/SensitiveDetector.cc app/SensitiveHit.cc
  app/TrackingAction.cc)
if(CELERITAS_USE_ROOT)
  list(APPEND APP_SOURCES app/RootIO.cc)
endif()
add_executable(m2fs-celer-g4 ${APP_SOURCES} CartMapIO.cc EventJson.cc)
target_include_directories(m2fs-celer-g4 PRIVATE . app ${Geant4_INCLUDE_DIRS})
celeritas_target_link_libraries(m2fs-celer-g4 Celeritas::accel Celeritas::ExtMPI
  nlohmann_json::nlohmann_json ${Geant4_LIBRARIES})
if(CELERITAS_USE_ROOT)
  celeritas_target_link_libraries(m2fs-celer-g4 ROOT::Tree)
endif()
add_executable(m2fs-field-check field-check.cc CartMapIO.cc)
target_include_directories(m2fs-field-check PRIVATE . ${Geant4_INCLUDE_DIRS})
celeritas_target_link_libraries(m2fs-field-check Celeritas::accel
  nlohmann_json::nlohmann_json ${Geant4_LIBRARIES})
