#ifndef CALO_REVIEW_MANIFEST_METADATA_HH
#define CALO_REVIEW_MANIFEST_METADATA_HH
#include "fhiclcpp/ParameterSet.h"
#include "cetlib_except/exception.h"
#include "TTree.h"
#include <algorithm>
#include <string>
namespace mu2e { namespace manifest_review {
// Claims, not proofs: an external launcher must hash the actual input bytes.
struct Metadata {
  std::string sourceArtifact, sourceSha256, sourceSnapshotSha256, expandedFclSha256;
  unsigned schemaVersion{1};
  explicit Metadata(fhicl::ParameterSet const& p)
    : sourceArtifact(p.get<std::string>("sourceArtifact")),
      sourceSha256(p.get<std::string>("sourceSha256")),
      sourceSnapshotSha256(p.get<std::string>("sourceSnapshotSha256")),
      expandedFclSha256(p.get<std::string>("expandedFclSha256")) {
    if (sourceArtifact.empty()) throw cet::exception("MANIFEST_CONFIG") << "empty sourceArtifact";
    for (auto const* s : {&sourceSha256, &sourceSnapshotSha256, &expandedFclSha256}) {
      if (s->size() != 64 || !std::all_of(s->begin(), s->end(), [](char c) {
          return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }))
        throw cet::exception("MANIFEST_CONFIG") << "SHA256 must be 64 lowercase hex digits";
    }
  }
  void branches(TTree* t) {
    t->Branch("schema_version", &schemaVersion);
    t->Branch("source_artifact", &sourceArtifact); t->Branch("source_sha256", &sourceSha256);
    t->Branch("source_snapshot_sha256", &sourceSnapshotSha256);
    t->Branch("expanded_fcl_sha256", &expandedFclSha256);
  }
};
} }
#endif
