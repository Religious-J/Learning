#include <iostream>
#include <memory>

class Myclass {
public:
  Myclass() { std::cout << "Create!!" << std::endl; }
  ~Myclass() { std::cout << "Over!!" << std::endl; }

  void dosomething() { std::cout << "Doing..." << std::endl; }
};

int main() {

  std::unique_ptr<Myclass> ptr(new Myclass());
  ptr->dosomething();

  //  通过 move 语义转移所有权
  std::unique_ptr<Myclass> anotherPtr = std::move(ptr);
  anotherPtr->dosomething();

  // 此时 ptr 已经是 nullptr
  if (!ptr) {
    std::cout << "prt is nullptr" << std::endl;
  }

  return 0;
}
