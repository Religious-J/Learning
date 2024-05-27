#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>

std::mutex mtx;

int sharedData = 0;

void increase() {

  for (int i = 0; i < 1000; i++) {
    std::lock_guard<std::mutex> lock(mtx);
    sharedData++;
  }
}

int main() {
  std::thread thread1(increase);

  std::thread thread2(increase);

  thread1.join();
  thread2.join();

  std::cout << "answer: " << sharedData << std::endl;

  return 0;
}
