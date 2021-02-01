#include <functional>
#include <memory>
#include <utility>

#include "field.hpp"

template <typename T>
class DynamicParameter {
 public:
  DynamicParameter() : dynamicField_(nullptr){};
  DynamicParameter(const DynamicParameter&);
  DynamicParameter& operator=(const DynamicParameter&);
  DynamicParameter(const DynamicParameter&&) noexcept;
  DynamicParameter& operator=(const DynamicParameter&&) noexcept;

  virtual ~DynamicParameter() = default;
  /** Add time-dependent function that is the same for every grid cell.
   * The input parameter value will be copied.
   *
   * Parameter values will be evaluated as:
   * a) uniform_value + term(t)
   * b) cell_value + term(t)
   *
   * @param term time-dependent function.
   */
  void addTimeDependentTerm(const std::function<T(real)>& term) {
    time_dep_terms.emplace_back(std::function<T(real)>(term), Field());
  }
  /** Add time-dependent function that is the same for every grid cell.
   * The input parameter values will be copied.
   *
   * Parameter values will be evaluated as:
   * a) uniform_value + term(t) * mask
   * b) cell_value + term(t) * cell_mask_value
   *
   * @param term time-dependent function.
   * @param mask define how the magnitude of the time-dependent function should
   *             depend on cell coordinates. The input value will be copied.
   */
  void addTimeDependentTerm(const std::function<T(real)>& term,
                            const Field& mask) {
    time_dep_terms.emplace_back(std::function<T(real)>(term), Field(mask));
  }
  /** Remove all time-dependet terms and their masks. */
  void removeAllTimeDependentTerms() { time_dep_terms.clear(); }
  /** Return true if parameter has non-zero time dependent terms. */
  bool isDynamic() const noexcept { return !time_dep_terms.empty(); };
  /** Return true if dynamic parameter values are independent of the cell
   * location. */
  bool isUniform() const;

 protected:
  /** Store time-dependent field values to be used on device. */
  mutable std::unique_ptr<Field> dynamicField_;
  /** List of all time dependent terms. */
  std::vector<std::pair<std::function<T(real)>, Field>> time_dep_terms;
  /** Evaluate time-dependent terms and add their values.
   *
   * @param t time to get values at.
   * @param p output value defined for the current system.
   */
  void evalTimeDependentTerms(real t, Field& p) const;
};