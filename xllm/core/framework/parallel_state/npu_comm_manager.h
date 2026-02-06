#include <acl/acl.h>
#include <atb/types.h>
#include <glog/logging.h>

#include <map>

#include "hccl/hccl.h"

namespace xllm {

const std::string LCCL = "lccl";
const std::string HCCL = "hccl";
const std::string LCOC = "lcoc";

/// A cache object contains information of a communication group
class NPUCommInfo {
 public:
  ~NPUCommInfo();

  uint64_t cacheId_ = 0;
  uint32_t subCommRankId_ = 0;
  std::vector<uint32_t> rankIds_ = {};
  std::string backend_ = "";
  HcclComm hcclComm_ = nullptr;
  uint32_t bufferSize_ = 0;
  uint32_t streamId_ = 0;
  bool enableReuse_ = true;

  std::string ToString() const;
};

/// A class manages all the communication group (including commDomain and
/// hcclComm ptr)
class NPUExternalCommManager {
 public:
  NPUExternalCommManager(HcclComm globalComm,
                         uint32_t worldSize,
                         uint32_t local_rank,
                         std::string backend,
                         std::string rankTableFile = "",
                         uint32_t streamId = 0);
  void SetLcclCommDomainRange(int32_t lowerBound, int32_t upperBound);

  void Reset();

  std::string GetCommDomain(uint32_t groupId,
                            const std::vector<uint32_t>& rankIds,
                            uint32_t subCommRankId,
                            std::string backend,
                            uint32_t bufferSize,
                            uint32_t streamId,
                            bool enableReuse = true);

  HcclComm GetCommPtr(std::string commDomain);

  std::shared_ptr<NPUCommInfo> GetCommInfo(std::string commDomain);

  bool IsInitialized();

  std::string PrintCommInfo();

  uint32_t worldSize_ = 0;
  uint32_t rank_;
  std::string rankTableFile_ = "";

 private:
  std::string GetCommDomainFromCache(const std::vector<uint32_t>& rankIds,
                                     std::string backend,
                                     uint32_t bufferSize,
                                     uint32_t streamId);
  std::string GetSelfAssignedCommDomain(std::shared_ptr<NPUCommInfo>& commInfo,
                                        uint32_t groupId);
  std::string GetHcclSubCommDomain(std::shared_ptr<NPUCommInfo>& commInfo,
                                   uint32_t groupId);

  std::map<std::string, std::shared_ptr<NPUCommInfo>> commInfoCache_ = {};
  HcclComm globalComm_ = nullptr;
  uint32_t commDomainCounter_ = 0;
  uint32_t lcclCommDomainLowerBound_ = 0;
  uint32_t lcclCommDomainUpperBound_ = 0;
};

}  // namespace xllm
