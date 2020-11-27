#pragma once

#include <memory>

template <class T>
class RefLink {
 public:
  explicit RefLink(T* object) : object_(object) {}
  T* get() { return object_; }

 private:
  T* object_;
};

template <class T>
class Ref {
 public:
  Ref() {}
  explicit Ref(const Ref<T>& ref) : link_(ref.link_) {}
  explicit Ref(std::shared_ptr<RefLink<T>> link) : link_(link) {}

  Ref& operator=(const Ref& other) {
    link_ = other.link_;
    return *this;
  }

  T* get() const {
    if (link_) return link_->get();
    return nullptr;
  }

  T* operator->() const { return get(); }

 private:
  std::shared_ptr<RefLink<T>> link_;
};

template <class T>
class RefHandler {
 public:
  explicit RefHandler(T* object)
      : link_(new RefLink<T>(object)), clink_(new RefLink<const T>(object)) {}
  Ref<T> ref() const { return Ref<T>(link_); }
  Ref<const T> constref() const { return Ref<const T>(clink_); }

 private:
  std::shared_ptr<RefLink<T>> link_;
  std::shared_ptr<RefLink<const T>> clink_;
};

//
//  RefHandler<Ferromagnet> refHandler{this};
//  Ref<Ferromagnet> ref() { return refHandler.ref(); }
//  Ref<const Ferromagnet> ref() const { return refHandler.constref(); }
