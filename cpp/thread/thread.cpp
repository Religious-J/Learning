#include <iostream>
#include <thread>

void printhello() { std::cout << "Hello nihao" << std::endl; }

int main() {

  std::thread thread1(printhello);

  std::thread thread2(printhello);

  thread1.join();

  thread2.join();

  std::cout << "end" << std::endl;

  return 0;
}
