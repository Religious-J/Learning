#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

std::mutex mtx;
std::condition_variable cv;
std::queue<int> tasks;

void work() {
  while (true) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [] { return !tasks.empty(); });
    int task = tasks.front();
    tasks.pop();
    lock.unlock();
    // do
    std::cout << "Task: " << task << " processed by thread "
              << std::this_thread::get_id() << std::endl;
  }
}

int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < 5; i++) {
    threads.push_back(std::thread(work));
  }

  // 添加任务
  for (int i = 0; i < 10; ++i) {
    std::lock_guard<std::mutex> lock(mtx);
    tasks.push(i);
    cv.notify_one(); // 唤醒一个等待的线程来获取任务
  }

  for (auto &thread : threads) {
    thread.join();
  }

  return 0;
}
