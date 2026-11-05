#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Core/FileBlock.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Provenance.h"
#include "art/Framework/Principal/SubRun.h"
#include "canvas/Persistency/Common/Ptr.h"
#include "canvas/Persistency/Provenance/ProcessHistoryID.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSetID.h"
#include "fhiclcpp/types/Atom.h"
#include "Offline/MCDataProducts/inc/GenEventCount.hh"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StatusG4.hh"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// This uncompiled proposal reads art products and never certifies persistence or exposure.
namespace mu2e {
namespace {
using U64 = std::uint64_t;

[[noreturn]] void fail(std::string const& message) {
  throw std::runtime_error("TargetReadback: " + message);
}

// Each row is bounded before appending, including its envelope and terminating LF.
class Json {
public:
  explicit Json(U64 limit) : limit_(limit) {}

  Json& raw(std::string const& text) {
    if (text.size() > limit_ - text_.size()) fail("maxBytes exceeded");
    text_ += text;
    return *this;
  }

  Json& string(std::string const& text) {
    raw("\"");
    static char const hex[] = "0123456789abcdef";
    for (std::size_t i = 0; i < text.size();) {
      unsigned char const c = static_cast<unsigned char>(text[i]);
      if (c == '"') { raw("\\\""); ++i; }
      else if (c == '\\') { raw("\\\\"); ++i; }
      else if (c < 0x20) {
        std::string escape = "\\u00";
        escape += hex[c >> 4];
        escape += hex[c & 15];
        raw(escape);
        ++i;
      } else if (c < 0x80) {
        raw(text.substr(i++, 1));
      } else {
        std::size_t n = 0;
        std::uint32_t point = 0;
        if (c >= 0xc2 && c <= 0xdf) { n = 2; point = c & 0x1f; }
        else if (c >= 0xe0 && c <= 0xef) { n = 3; point = c & 0x0f; }
        else if (c >= 0xf0 && c <= 0xf4) { n = 4; point = c & 0x07; }
        else fail("invalid UTF-8 in JSON string");
        if (n > text.size() - i) fail("truncated UTF-8 in JSON string");
        for (std::size_t j = 1; j < n; ++j) {
          unsigned char const next = static_cast<unsigned char>(text[i + j]);
          if ((next & 0xc0) != 0x80) fail("invalid UTF-8 continuation");
          point = (point << 6) | (next & 0x3f);
        }
        if ((n == 2 && point < 0x80) || (n == 3 && point < 0x800) ||
            (n == 4 && point < 0x10000) || point > 0x10ffff ||
            (point >= 0xd800 && point <= 0xdfff)) fail("invalid UTF-8 scalar");
        raw(text.substr(i, n));
        i += n;
      }
    }
    return raw("\"");
  }

  Json& integer(U64 value) { return raw(std::to_string(value)); }
  Json& signedInteger(int value) { return raw(std::to_string(value)); }
  Json& boolean(bool value) { return raw(value ? "true" : "false"); }
  Json& real(double value) {
    if (!std::isfinite(value)) fail("nonfinite product value");
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return raw(stream.str());
  }
  std::string const& text() const { return text_; }

private:
  U64 limit_;
  std::string text_;
};

void requireTag(art::InputTag const& tag) {
  if (tag.label().empty() || tag.process().empty() ||
      tag.process() == "@current_process" || tag.process() == "@skipCurrentProcess" ||
      tag.process().find_first_of("@*?:") != std::string::npos)
    fail("all four InputTags require a literal nonempty module label and explicit process");
}

template <typename Handle>
void provenance(Json& json, Handle const& handle, art::InputTag const& tag) {
  auto const* p = handle.provenance();
  if (!p || !p->isValid() || !p->isPresent() ||
      p->productID() != handle.id() || handle.id().value() == 0)
    fail("missing or inconsistent product provenance");
  if (p->moduleLabel() != tag.label() || p->productInstanceName() != tag.instance() ||
      p->processName() != tag.process()) fail("provenance differs from explicit InputTag");
  if (p->branchName().empty() || p->friendlyClassName().empty() || p->psetIDs().empty())
    fail("incomplete provenance");
  json.raw("{\"branch_name\":").string(p->branchName())
    .raw(",\"module_label\":").string(p->moduleLabel())
    .raw(",\"product_instance_name\":").string(p->productInstanceName())
    .raw(",\"process_name\":").string(p->processName())
    .raw(",\"friendly_class_name\":").string(p->friendlyClassName())
    .raw(",\"parameter_set_ids\":[");
  bool first = true;
  for (auto const& id : p->psetIDs()) {
    if (!id.is_valid() || id.to_string().empty()) fail("invalid provenance parameter-set ID");
    if (!first) json.raw(",");
    first = false;
    json.string(id.to_string());
  }
  json.raw("]}");
}

using GenHandle = art::ValidHandle<GenParticleCollection>;
using SimHandle = art::ValidHandle<SimParticleCollection>;

void verify(art::Ptr<GenParticle> const& ptr, GenHandle const& handle) {
  if (ptr.isNull()) return;
  if (ptr.id() != handle.id() || ptr.key() >= handle->size())
    fail("generator Ptr is outside the loaded generator handle");
  if (!ptr.isAvailable()) fail("generator Ptr is unavailable");
  if (ptr.get() != &handle->at(ptr.key())) fail("generator Ptr resolved-object identity mismatch");
}

void verify(art::Ptr<SimParticle> const& ptr, SimHandle const& handle) {
  if (ptr.isNull()) return;
  if (ptr.id() != handle.id()) fail("SimParticle Ptr is outside the loaded sim handle");
  auto const found = handle->find(SimParticle::key_type(ptr.key()));
  if (found == handle->end() || found->first.asUint() != ptr.key())
    fail("SimParticle Ptr key is absent from the actual map-vector");
  if (!ptr.isAvailable()) fail("SimParticle Ptr is unavailable");
  if (ptr.get() != &found->second) fail("SimParticle Ptr resolved-object identity mismatch");
}

template <typename Particle, typename Handle>
void pointer(Json& json, art::Ptr<Particle> const& ptr, Handle const& handle) {
  if (ptr.isNull()) { json.raw("null"); return; }
  verify(ptr, handle);
  json.raw("{\"product_id\":").integer(ptr.id().value())
    .raw(",\"key\":").integer(ptr.key()).raw("}");
}

// This owner closes resources without emitting a completion row from its destructor.
class Output {
public:
  Output() = default;
  Output(Output const&) = delete;
  Output& operator=(Output const&) = delete;
  ~Output() { if (file_) std::fclose(file_); }

  void open(std::string const& path) {
    if (::mkdir(path.c_str(), 0700) != 0)
      fail("cannot exclusively create outputDirectory: " + std::string(std::strerror(errno)));
    int const directory = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (directory < 0) fail("cannot open newly created outputDirectory");
    pending_ = path + "/readback.pending.jsonl";
    final_ = path + "/readback.jsonl";
    int const fd = ::openat(directory, "readback.pending.jsonl",
                            O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    int const openError = errno;
    int const closeResult = ::close(directory);
    if (fd < 0) fail("cannot exclusively create readback.jsonl: " + std::string(std::strerror(openError)));
    if (closeResult != 0) { ::close(fd); fail("output directory close failed"); }
    file_ = ::fdopen(fd, "wb");
    if (!file_) { ::close(fd); fail("fdopen failed for readback.jsonl"); }
  }

  void write(std::string const& row) {
    if (!file_) fail("output is not open");
    if (std::fwrite(row.data(), 1, row.size(), file_) != row.size() ||
        std::fflush(file_) != 0 || std::ferror(file_)) fail("JSONL write/flush failed");
  }

  void close() {
    if (!file_) fail("output is not open at endJob");
    if (std::fflush(file_) != 0 || std::ferror(file_)) fail("final JSONL flush failed");
    auto* file = file_;
    file_ = nullptr;
    if (std::fclose(file) != 0) fail("final JSONL close failed");
    if (::link(pending_.c_str(), final_.c_str()) != 0) fail("exclusive final JSONL publication failed");
  }

private:
  std::FILE* file_ = nullptr;
  std::string pending_, final_;
};
}

// The sequential EDAnalyzer has one active input file and rejects duplicate global identities.
class TargetReadback : public art::EDAnalyzer {
public:
  struct Config {
    fhicl::Atom<art::InputTag> generatorTag{fhicl::Name("generatorTag")};
    fhicl::Atom<art::InputTag> simTag{fhicl::Name("simTag")};
    fhicl::Atom<art::InputTag> statusTag{fhicl::Name("statusTag")};
    fhicl::Atom<art::InputTag> countTag{fhicl::Name("countTag")};
    fhicl::Atom<std::string> outputDirectory{fhicl::Name("outputDirectory")};
    fhicl::Atom<U64> maxRows{fhicl::Name("maxRows")};
    fhicl::Atom<U64> maxBytes{fhicl::Name("maxBytes")};
    fhicl::Atom<U64> maxEvents{fhicl::Name("maxEvents")};
    fhicl::Atom<U64> maxParticlesPerEvent{fhicl::Name("maxParticlesPerEvent")};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;

  explicit TargetReadback(Parameters const& config)
    : art::EDAnalyzer(config), generatorTag_(config().generatorTag()),
      simTag_(config().simTag()), statusTag_(config().statusTag()), countTag_(config().countTag()),
      outputDirectory_(config().outputDirectory()), maxRows_(config().maxRows()),
      maxBytes_(config().maxBytes()), maxEvents_(config().maxEvents()),
      maxParticles_(config().maxParticlesPerEvent()) {
    requireTag(generatorTag_);
    requireTag(simTag_);
    requireTag(statusTag_);
    requireTag(countTag_);
    if (!maxRows_ || !maxBytes_ || !maxEvents_ || !maxParticles_)
      fail("all row/byte/event/particle caps must be positive");
    if (maxBytes_ > std::string().max_size()) fail("maxBytes exceeds string capacity");
    std::filesystem::path const path(outputDirectory_);
    if (outputDirectory_.empty() || outputDirectory_.find('\0') != std::string::npos ||
        !path.is_absolute() || path.filename().empty() || path.filename() == "." || path.filename() == "..")
      fail("outputDirectory must name a new absolute directory");
    consumes<GenParticleCollection>(generatorTag_);
    consumes<SimParticleCollection>(simTag_);
    consumes<StatusG4>(statusTag_);
    consumes<GenEventCount, art::InSubRun>(countTag_);
  }

private:
  void beginJob() override {
    guarded([&] {
      if (begun_) fail("duplicate beginJob");
      output_.open(outputDirectory_);
      row("reader_begin", [&](Json& j) {
        j.raw("{\"generator_tag\":").string(generatorTag_.encode())
          .raw(",\"sim_tag\":").string(simTag_.encode())
          .raw(",\"status_tag\":").string(statusTag_.encode())
          .raw(",\"count_tag\":").string(countTag_.encode())
          .raw(",\"max_rows\":").integer(maxRows_)
          .raw(",\"max_bytes\":").integer(maxBytes_)
          .raw(",\"max_events\":").integer(maxEvents_)
          .raw(",\"max_particles_per_event\":").integer(maxParticles_).raw("}");
      });
      begun_ = true;
    });
  }

  void respondToOpenInputFile(art::FileBlock const& block) override {
    guarded([&] {
      requireRunning();
      auto const& name = block.fileName();
      if (!activeFile_.empty() || name.empty()) fail("nested or empty input filename");
      if (!files_.insert(name).second) fail("reopened input filename");
      row("file_open", [&](Json& j) { j.raw("{\"input_file_name\":").string(name).raw("}"); });
      activeFile_ = name;
      ++fileCount_;
    });
  }

  void respondToCloseInputFile(art::FileBlock const& block) override {
    guarded([&] {
      requireActive();
      if (block.fileName() != activeFile_) fail("input close filename differs from active file");
      for (auto const& item : subrunFiles_)
        if (item.second == activeFile_ && !subruns_.count(item.first)) fail("input close before SubRun closure");
      row("file_close", [&](Json& j) { j.raw("{\"input_file_name\":").string(activeFile_).raw("}"); });
      activeFile_.clear();
    });
  }

  void analyze(art::Event const& event) override {
    guarded([&] {
      requireActive();
      if (eventCount_ >= maxEvents_) fail("maxEvents exceeded");
      std::array<U64, 3> const id{{event.id().run(), event.id().subRun(), event.id().event()}};
      if (!events_.insert(id).second) fail("duplicate global EventID");
      std::pair<U64, U64> const subrun{id[0], id[1]};
      if (subruns_.count(subrun)) fail("event after SubRun closure");
      auto const owner = subrunFiles_.emplace(subrun, activeFile_);
      if (owner.first->second != activeFile_) fail("SubRun split across files is unsupported");
      ++subrunEvents_[subrun];
      auto const generators = event.getValidHandle<GenParticleCollection>(generatorTag_);
      auto const sims = event.getValidHandle<SimParticleCollection>(simTag_);
      auto const status = event.getValidHandle<StatusG4>(statusTag_);
      // The particle cap applies separately to each actual product collection.
      if (generators->size() > maxParticles_ || sims->size() > maxParticles_)
        fail("maxParticlesPerEvent exceeded");
      std::ostringstream history;
      history.imbue(std::locale::classic());
      history << event.processHistoryID();
      if (history.str().empty()) fail("empty process history ID");
      row("event", [&](Json& j) {
        j.raw("{\"input_file_name\":").string(activeFile_)
          .raw(",\"event_id\":[").integer(id[0]).raw(",").integer(id[1]).raw(",").integer(id[2])
          .raw("],\"process_history_id\":").string(history.str())
          .raw(",\"generator_product_id\":").integer(generators.id().value())
          .raw(",\"generator_provenance\":");
        provenance(j, generators, generatorTag_);
        j.raw(",\"generated_primaries\":[");
        for (std::size_t key = 0; key < generators->size(); ++key) {
          if (key) j.raw(",");
          auto const& g = generators->at(key);
          j.raw("{\"generator_product_id\":").integer(generators.id().value())
            .raw(",\"generator_key\":").integer(key)
            .raw(",\"pdg\":").signedInteger(static_cast<int>(g.pdgId()))
            .raw(",\"generator_id\":").signedInteger(static_cast<int>(g.generatorId().id()))
            .raw(",\"position_mu2e_mm\":[").real(g.position().x()).raw(",").real(g.position().y())
            .raw(",").real(g.position().z()).raw("]")
            .raw(",\"four_momentum_MeV\":[").real(g.momentum().px()).raw(",").real(g.momentum().py())
            .raw(",").real(g.momentum().pz()).raw(",").real(g.momentum().e()).raw("]")
            .raw(",\"global_time_ns\":").real(g.time())
            .raw(",\"proper_time_ns\":").real(g.properTime()).raw(",\"generator_weight\":null}");
        }
        j.raw("],\"sim_product_id\":").integer(sims.id().value()).raw(",\"sim_provenance\":");
        provenance(j, sims, simTag_);
        j.raw(",\"sim_particles\":[");
        bool first = true;
        for (auto const& entry : *sims) {
          if (!first) j.raw(",");
          first = false;
          auto const& p = entry.second;
          j.raw("{\"product_id\":").integer(sims.id().value())
            .raw(",\"key\":").integer(entry.first.asUint())
            .raw(",\"object_key\":").integer(p.id().asUint())
            .raw(",\"sim_stage\":").integer(p.simStage())
            .raw(",\"pdg\":").signedInteger(static_cast<int>(p.pdgId()))
            .raw(",\"creation_code\":").signedInteger(static_cast<int>(p.creationCode().id()))
            .raw(",\"parent\":");
          pointer(j, p.parent(), sims);
          j.raw(",\"generator\":");
          pointer(j, p.genParticle(), generators);
          j.raw(",\"daughters\":[");
          bool firstDaughter = true;
          if (p.daughters().size() > maxParticles_) fail("daughter cap exceeded");
          for (auto const& daughter : p.daughters()) {
            if (daughter.isNull()) fail("null daughter Ptr");
            if (!firstDaughter) j.raw(",");
            firstDaughter = false;
            pointer(j, daughter, sims);
          }
          j.raw("]}");
        }
        j.raw("],\"status_product_id\":").integer(status.id().value()).raw(",\"status_provenance\":");
        provenance(j, status, statusTag_);
        j.raw(",\"status\":{\"status_native\":").signedInteger(status->status())
          .raw(",\"sim_overflow\":").boolean(status->overflowSimParticles())
          .raw(",\"step_kills\":").signedInteger(status->nKilledStepLimit())
          .raw(",\"field_kills\":").signedInteger(status->nKilledByFieldPropagator())
          .raw(",\"n_g4_tracks\":").signedInteger(status->nG4Tracks())
          .raw("},\"links_resolved_in_readback\":true,\"persistent_output_verified\":false}");
      });
      ++eventCount_;
    });
  }

  void endSubRun(art::SubRun const& subrun) override {
    guarded([&] {
      requireActive();
      std::pair<U64, U64> const id{subrun.id().run(), subrun.id().subRun()};
      auto const owner = subrunFiles_.find(id);
      if (owner == subrunFiles_.end() || owner->second != activeFile_) fail("eventless or wrong-file SubRun");
      if (!subruns_.insert(id).second) fail("duplicate global SubRun identity");
      auto const count = subrun.getValidHandle<GenEventCount>(countTag_);
      if (count->count() != subrunEvents_.at(id)) fail("SubRun callback/written event count mismatch");
      row("subrun", [&](Json& j) {
        j.raw("{\"input_file_name\":").string(activeFile_)
          .raw(",\"run\":").integer(id.first).raw(",\"subrun\":").integer(id.second)
          .raw(",\"count_product_id\":").integer(count.id().value()).raw(",\"count_provenance\":");
        provenance(j, count, countTag_);
        j.raw(",\"generated_event_callbacks\":").integer(count->count()).raw("}");
      });
      ++subrunCount_;
    });
  }

  // endJob refuses an active file rather than assuming an unverified callback ordering.
  void endJob() override {
    guarded([&] {
      requireRunning();
      if (!activeFile_.empty()) fail("endJob reached before input file close callback");
      if (files_.empty() || subruns_.size() != subrunFiles_.size()) fail("incomplete input/SubRun set");
      row("reader_end", [&](Json& j) {
        j.raw("{\"event_count\":").integer(eventCount_)
          .raw(",\"subrun_count\":").integer(subrunCount_)
          .raw(",\"file_count\":").integer(fileCount_)
          .raw(",\"readback_callback_complete\":true,\"framework_job_success_verified\":false,")
          .raw("\"persistent_output_verified\":false,\"exposure_certified\":false}");
      });
      output_.close();
      complete_ = true;
    });
  }

  template <typename Action>
  void guarded(Action action) {
    if (failed_) fail("reader is latched failed");
    try { action(); }
    catch (...) { failed_ = true; throw; }
  }

  void requireRunning() const {
    if (!begun_ || complete_) fail("callback outside active reader lifetime");
  }
  void requireActive() const {
    requireRunning();
    if (activeFile_.empty()) fail("event/subrun/close without active input file");
  }

  template <typename Data>
  void row(char const* kind, Data data) {
    if (rows_ >= maxRows_) fail("maxRows exceeded");
    Json json(maxBytes_ - bytes_);
    json.raw("{\"schema\":\"target-written-products-proposal-v1\",\"kind\":").string(kind)
      .raw(",\"data\":");
    data(json);
    json.raw(",\"proposal_only\":true,\"library_eligible\":false}\n");
    output_.write(json.text());
    bytes_ += json.text().size();
    ++rows_;
  }

  art::InputTag const generatorTag_, simTag_, statusTag_, countTag_;
  std::string const outputDirectory_;
  U64 const maxRows_, maxBytes_, maxEvents_, maxParticles_;
  Output output_;
  bool begun_ = false, failed_ = false, complete_ = false;
  U64 rows_ = 0, bytes_ = 0, eventCount_ = 0, subrunCount_ = 0, fileCount_ = 0;
  std::string activeFile_;
  std::set<std::string> files_;
  std::set<std::array<U64, 3>> events_;
  std::set<std::pair<U64, U64>> subruns_;
  std::map<std::pair<U64, U64>, std::string> subrunFiles_;
  std::map<std::pair<U64, U64>, U64> subrunEvents_;
};
}

DEFINE_ART_MODULE(mu2e::TargetReadback)
