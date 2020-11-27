#pragma once

#include <memory>
#include <vector>

template <class T>
class Handle;

template <class T>
class Handler;

template <class T>
struct HandlerLink {
  Handler<T>* handler;
};

template <class T>
class Handler {
 public:
  // Create an object handler. When the handler is destroyed, the object will be
  // destroyed
  explicit Handler(T* object) : ptr_(object), link_(new HandlerLink<T>{this}) {}

  // Move constructor
  Handler(Handler&& other) noexcept : ptr_(other.ptr_), link_(other.link_) {
    // update the handle pointer in the handlerlink
    link_->handler = this;

    // avoid the invalidation of the handlerlink when 'other' gets destroyed
    // after the move
    other.link_ = nullptr;

    // avoid deletion of the object when 'other' gets destroyed after the move
    other.ptr_ = nullptr;
  }

  // Destroy the handled object and make the handle link invalid
  ~Handler() {
    delete ptr_;
    if (link_)  // could point to nothing after a Handler move
      link_->handler = nullptr;
  }

  T* get() const { return ptr_; }
  T* operator->() const { return ptr_; }

  // Return pointer to the handlerlink. The handlerlink should only be modified
  // by the handler, so we return a ptr to a const
  std::shared_ptr<const HandlerLink<T>> link() const {
    return std::shared_ptr<const HandlerLink<T>>(link_);
  }

 private:
  // A handler should not be moved
  Handler(const Handler<T>& handler);
  Handler& operator=(const Handler& other);

 private:
  // Pointer to the object which is being handled by this handler
  T* ptr_;
  std::shared_ptr<HandlerLink<T>> link_;
};

template <class T>
class Handle {
 public:
  Handle() {}
  Handle(const Handle<T>& handle) : link_(handle.link_) {}
  explicit Handle(const Handler<T>& handler) : link_(handler.link()) {}

  Handle& operator=(const Handle& other) {
    link_ = other.link_;
    return *this;
  }

  Handle& operator=(const Handler<T>& handler) {
    link_ = handler.link_;
    return *this;
  }

  bool operator==(const Handle<T>& other) const { return get() == other.get(); }

  bool operator!=(const Handle<T>& other) const { return get() != other.get(); }

  // We might want to use map with a Handle as the key
  // TODO: find a less hacky approach
  bool operator<(const Handle& other) const {
    return link_.get() < other.link_.get();
  }

  // Returns a pointer to the handled object. This might be a nullptr if the
  // object is being deleted by the handler
  T* get() const {
    if (link_) return link_->handler->operator->();
    return nullptr;
  }

  T* operator->() const { return get(); }

  // Returns true if an object is handled, otherwise return false
  explicit operator bool() const { return link_ && link_->handler != nullptr; }

 private:
  std::shared_ptr<const HandlerLink<T>> link_;
};
