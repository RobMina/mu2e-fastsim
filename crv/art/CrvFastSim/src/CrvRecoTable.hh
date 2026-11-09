// Reader shared by the *FromTable producers: one tree of a CrvRecoTableDumper-layout file (directory crvdump/ by
// default), rows grouped by (run, subrun, event) as row-index lists. Jagged columns may be std::vector branches (the
// art dumper) or counted C arrays (uproot writer, counter n<name>); Jagged<T> reads either.
#ifndef CrvFastSim_CrvRecoTable_hh
#define CrvFastSim_CrvRecoTable_hh

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "TBranch.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TTree.h"
#include "cetlib_except/exception.h"

namespace mu2e {
namespace crvfastsim {

using EventKey = std::tuple<unsigned, unsigned, unsigned>;

class RecoTable {
 public:
  RecoTable(const std::string& file, const std::string& dir, const std::string& tree) {
    file_.reset(TFile::Open(file.c_str()));
    if (!file_ || file_->IsZombie()) throw cet::exception("CRVFASTSIM") << "cannot open " << file;
    auto* d = file_->GetDirectory(dir.c_str());
    tree_ = d ? d->Get<TTree>(tree.c_str()) : nullptr;
    if (!tree_) throw cet::exception("CRVFASTSIM") << "no " << dir << "/" << tree << " TTree in " << file;
    UInt_t run, sub, evt;
    tree_->SetBranchStatus("*", false);
    for (const char* b : {"run", "subrun", "event"}) tree_->SetBranchStatus(b, true);
    tree_->SetBranchAddress("run", &run);
    tree_->SetBranchAddress("subrun", &sub);
    tree_->SetBranchAddress("event", &evt);
    for (Long64_t i = 0; i < tree_->GetEntries(); ++i) {
      tree_->GetEntry(i);
      rows_[EventKey(run, sub, evt)].push_back(i);
    }
    tree_->ResetBranchAddresses();
    tree_->SetBranchStatus("*", true);
  }
  TTree* tree() { return tree_; }
  // row indices of one event, in file order (empty if the event has no rows)
  const std::vector<Long64_t>& rows(const EventKey& k) const {
    auto it = rows_.find(k);
    return it == rows_.end() ? empty_ : it->second;
  }

 private:
  std::unique_ptr<TFile> file_;
  TTree* tree_{nullptr};
  std::map<EventKey, std::vector<Long64_t>> rows_;
  std::vector<Long64_t> empty_;
};

// one jagged column of a tree: set up once, then values() after each GetEntry
template <class T>
class Jagged {
 public:
  Jagged(TTree* t, const char* name) {
    TBranch* b = t->GetBranch(name);
    if (!b) throw cet::exception("CRVFASTSIM") << "no branch " << name;
    if (std::string(b->GetClassName()).find("vector") != std::string::npos) {
      vec_ = true;
      t->SetBranchAddress(name, &vp_);
      return;
    }
    TLeaf* l = b->GetLeaf(name);
    TLeaf* c = l ? l->GetLeafCount() : nullptr;
    if (!c) throw cet::exception("CRVFASTSIM") << "branch " << name << " is neither a vector nor a counted array";
    double mx = t->GetEntries() ? t->GetMaximum(c->GetName()) : 0.;
    buf_.resize(static_cast<size_t>(mx > 0 ? mx : 0) + 1);
    t->SetBranchAddress(c->GetName(), &n_);
    t->SetBranchAddress(name, buf_.data());
  }
  Jagged(const Jagged&) = delete;
  Jagged& operator=(const Jagged&) = delete;
  std::vector<T> values() const {
    return vec_ ? v_ : std::vector<T>(buf_.begin(), buf_.begin() + n_);
  }

 private:
  bool vec_{false};
  std::vector<T> v_;
  std::vector<T>* vp_{&v_};
  std::vector<T> buf_;
  Int_t n_{0};
};

}  // namespace crvfastsim
}  // namespace mu2e

#endif
