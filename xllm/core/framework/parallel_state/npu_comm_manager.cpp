#include "npu_comm_manager.h"

#include <cerrno>
#include <cmath>

#include "securec.h"

namespace xllm {

std::string RanktoString(const std::vector<uint32_t>& rankIds) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < rankIds.size(); ++i) {
    oss << rankIds[i];
    if (i != rankIds.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

NPUCommInfo::~NPUCommInfo() {
  LOG(INFO) << "External Comm Manager: NPUCommInfo ["
            << std::hash<const NPUCommInfo*>{}(this) << "] destruction starts.";
  if (hcclComm_ != nullptr) {
    auto ret = HcclCommDestroy(hcclComm_);
    if (ret != HCCL_SUCCESS) {
      LOG(ERROR)
          << "External Comm Manager: Call `HcclCommDestroy` API from CANN "
          << "to destroy hccl communication group failed. "
          << "Error code: " << ret << ". "
          << "Check the default log path at $HOME/ascend/log for more "
             "details. ";
    }
  }
  hcclComm_ = nullptr;
}

std::string NPUCommInfo::ToString() const {
  auto ranks = RanktoString(rankIds_);
  std::stringstream ss;
  ss << "Cache Addr[" << this << "] cacheId_: " << cacheId_
     << ", subCommRankId_: " << subCommRankId_ << ", rankIds_: " << ranks
     << ", bufferSize_: " << bufferSize_ << ", backend_: " << backend_
     << ", hcclComm_: " << hcclComm_ << ", streamId_: " << streamId_;
  return ss.str();
}

bool AreVectorsEqual(const std::vector<uint32_t>& rankIdsA,
                     const std::vector<uint32_t>& rankIdsB) {
  if (rankIdsA.size() != rankIdsB.size()) {
    return false;
  }
  for (size_t i = 0; i < rankIdsA.size(); i++) {
    if (rankIdsA.at(i) != rankIdsB.at(i)) {
      return false;
    }
  }
  return true;
}

NPUExternalCommManager::NPUExternalCommManager(HcclComm globalComm,
                                               uint32_t worldSize,
                                               uint32_t local_rank,
                                               std::string backend,
                                               std::string rankTableFile,
                                               uint32_t streamId)
    : worldSize_(worldSize),
      rank_(local_rank),
      rankTableFile_(rankTableFile),
      globalComm_(globalComm) {
  std::vector<uint32_t> rankIds = {};
  for (uint32_t id = 0; id < worldSize; id++) {
    rankIds.push_back(id);
  }
  std::shared_ptr<NPUCommInfo> commInfo = std::make_shared<NPUCommInfo>();
  commInfo->cacheId_ = commInfoCache_.size();
  commInfo->subCommRankId_ = local_rank;
  commInfo->rankIds_ = rankIds;
  commInfo->backend_ = backend;
  commInfo->streamId_ = streamId;
  if (this->globalComm_ == nullptr) {
    LOG(ERROR) << "External Comm Manager: Create the hccl communication group "
                  "failed, please make sure the globalComm was initialized";
  }
  commInfo->hcclComm_ = this->globalComm_;
  char hcclCommName[128] = {};
  HcclGetCommName(this->globalComm_, hcclCommName);
  auto commDomain = std::string(hcclCommName);
  LOG(INFO) << "External Comm Manager: Add [" << commDomain << "] to cache.";
  commInfoCache_[commDomain] = commInfo;
}

void NPUExternalCommManager::SetLcclCommDomainRange(int32_t lowerBound,
                                                    int32_t upperBound) {
  lcclCommDomainLowerBound_ = lowerBound;
  lcclCommDomainUpperBound_ = upperBound;
}

void NPUExternalCommManager::Reset() {
  worldSize_ = 0;
  rank_ = 0;
  rankTableFile_ = "";
  commDomainCounter_ = 0;
  lcclCommDomainLowerBound_ = 0;
  lcclCommDomainUpperBound_ = 0;
  globalComm_ = nullptr;
  commInfoCache_.clear();
}

std::string NPUExternalCommManager::GetCommDomain(
    uint32_t groupId,
    const std::vector<uint32_t>& rankIds,
    uint32_t subCommRankId,
    std::string backend,
    uint32_t bufferSize,
    uint32_t streamId,
    bool enableReuse) {
  auto ranks = RanktoString(rankIds);

  LOG(INFO) << "External Comm Manager: try to create comm with rankIds "
            << ranks << ", subCommRankId " << subCommRankId
            << ", backend: " << backend << ", bufferSize " << bufferSize
            << ", streamId " << streamId;

  std::string commDomain = "";

  if (rankIds.size() <= 1) {
    return commDomain;
  }

  if (enableReuse) {
    LOG(INFO) << "External Comm Manager: try to reuse communication group from "
                 "cache.";
    commDomain = GetCommDomainFromCache(rankIds, backend, bufferSize, streamId);
    if (commDomain != "") {
      return commDomain;
    }
  }

  std::shared_ptr<NPUCommInfo> commInfo = std::make_shared<NPUCommInfo>();
  commInfo->cacheId_ = commInfoCache_.size();
  commInfo->subCommRankId_ = subCommRankId;
  commInfo->rankIds_ = rankIds;
  commInfo->backend_ = backend;
  commInfo->bufferSize_ = bufferSize;
  commInfo->streamId_ = streamId;
  commInfo->enableReuse_ = enableReuse;
  if ((backend == LCCL || backend == LCOC) && rankIds.size() > 1) {
    commDomain = GetSelfAssignedCommDomain(commInfo, groupId);
  } else if (backend == HCCL && rankIds.size() > 1) {
    commDomain = GetHcclSubCommDomain(commInfo, groupId);
  }
  commInfoCache_[commDomain] = commInfo;
  LOG(INFO) << "External Comm Manager: Add [" << commDomain << "] to cache";
  return commDomain;
}

std::string NPUExternalCommManager::GetCommDomainFromCache(
    const std::vector<uint32_t>& rankIds,
    std::string backend,
    uint32_t bufferSize,
    uint32_t streamId) {
  std::map<std::string, std::shared_ptr<NPUCommInfo>>::iterator it;
  for (it = commInfoCache_.begin(); it != commInfoCache_.end(); it++) {
    if (AreVectorsEqual(it->second->rankIds_, rankIds) &&
        it->second->backend_ == backend &&
        it->second->bufferSize_ == bufferSize &&
        it->second->streamId_ == streamId && it->second->enableReuse_) {
      auto ranks = RanktoString(rankIds);
      LOG(INFO) << "External Comm Manager: Comm with rankIds " << ranks
                << ", bufferSize " << bufferSize << ", backend: " << backend
                << ", streamId" << streamId << " hit. CommDomain [" << it->first
                << "] is reused.";
      return it->first;
    }
  }
  return "";
}

std::string NPUExternalCommManager::GetSelfAssignedCommDomain(
    std::shared_ptr<NPUCommInfo>& commInfo,
    uint32_t groupId) {
  uint32_t commDomainInt = 0;
  if ((lcclCommDomainLowerBound_ < UINT32_MAX - commDomainCounter_) &&
      (groupId < UINT32_MAX - commDomainCounter_ - lcclCommDomainLowerBound_)) {
    commDomainInt = lcclCommDomainLowerBound_ + commDomainCounter_ + groupId;
  } else {
    std::stringstream ss;
    ss << "External Comm Manager: overflow detected when counting commDomain "
          "index, "
       << "got lcclCommDomainLowerBound: " << lcclCommDomainLowerBound_ << ", "
       << "commDomainCounter_: " << commDomainCounter_ << ", "
       << "and groupId: " << groupId << ".";
    throw std::runtime_error(ss.str());
  }

  if (commDomainInt >= lcclCommDomainUpperBound_) {
    std::stringstream ss;
    ss << "External Comm Manager: Lccl commDomain exceeds the upper bound. "
       << "Available commDomain range is [" << lcclCommDomainLowerBound_ << ", "
       << lcclCommDomainUpperBound_ << "]. "
       << "The range of the communication domain is determinded by "
          "`num_lccl_comm_shards` "
       << "and `lccl_comm_shard_id`. Please review initializaion parameters "
       << "of the `GeneratorTorch` object.";
    throw std::runtime_error(ss.str());
  }
  std::string commDomain = std::to_string(commDomainInt);
  commDomainCounter_ =
      commDomainCounter_ + ceil(worldSize_ / commInfo->rankIds_.size());
  LOG(INFO) << "External Comm Manager: commDomainCounter_ update to "
            << commDomainCounter_;
  return commDomain;
}

std::string NPUExternalCommManager::GetHcclSubCommDomain(
    std::shared_ptr<NPUCommInfo>& commInfo,
    uint32_t groupId) {
  LOG(INFO) << "GetHcclSubCommDomain start.";
  std::string commDomain = "";
  if (globalComm_ != nullptr) {
    HcclComm hcclComm;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclBufferSize = commInfo->bufferSize_;
    std::vector<uint32_t> tempRankIds = {};
    for (auto item : commInfo->rankIds_) {
      tempRankIds.push_back(item);
    }
    auto ret = HcclCreateSubCommConfig(&globalComm_,
                                       tempRankIds.size(),
                                       tempRankIds.data(),
                                       commInfo->cacheId_,
                                       commInfo->subCommRankId_,
                                       &config,
                                       &hcclComm);
    if (hcclComm == nullptr) {
      LOG(ERROR) << "External Comm Manager: Call `HcclCreateSubCommConfig` API "
                    "from CANN "
                 << "to create the hccl communication group failed. "
                 << "Error code: " << ret << ". "
                 << "Check the default log path at $HOME/ascend/log for more "
                    "details. ";
    }
    commInfo->hcclComm_ = hcclComm;
    char hcclCommName[128] = {};
    HcclGetCommName(hcclComm, hcclCommName);
    commDomain = std::string(hcclCommName);
  } else {
    commDomain = GetSelfAssignedCommDomain(commInfo, groupId);
  }
  LOG(INFO) << "GetHcclSubCommDomain end.";
  return commDomain;
}

HcclComm NPUExternalCommManager::GetCommPtr(std::string commDomain) {
  if (commDomain == "") {
    return nullptr;
  }
  auto it = commInfoCache_.find(commDomain);
  if (it == commInfoCache_.end()) {
    std::stringstream ss;
    ss << "External Comm Manager: Comm domain[" << commDomain
       << "] not found in cache.";
    throw std::out_of_range(ss.str());
  }
  return it->second->hcclComm_;
}

std::shared_ptr<NPUCommInfo> NPUExternalCommManager::GetCommInfo(
    std::string commDomain) {
  if (commDomain == "") {
    return nullptr;
  }
  auto it = commInfoCache_.find(commDomain);
  if (it == commInfoCache_.end()) {
    std::stringstream ss;
    ss << "External Comm Manager: Comm domain[" << commDomain
       << "] not found in cache.";
    throw std::out_of_range(ss.str());
  }
  return it->second;
}

bool NPUExternalCommManager::IsInitialized() {
  return commInfoCache_.size() > 0;
}

std::string NPUExternalCommManager::PrintCommInfo() {
  std::stringstream ss;
  ss << "External Comm Manager: Comm Info Cache Summary: Count "
     << commInfoCache_.size();
  std::map<std::string, std::shared_ptr<NPUCommInfo>>::const_iterator it;
  for (it = commInfoCache_.begin(); it != commInfoCache_.end(); it++) {
    ss << " Comm domain[" << it->first << "] " << it->second->ToString();
  }
  return ss.str();
}

}  // namespace xllm
