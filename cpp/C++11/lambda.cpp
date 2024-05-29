#include <algorithm>
#include <iostream>
#include <vector>

int main() {
  std::vector<int> nums = {1, 2, 3, 4, 5};

  std::for_each(nums.begin(), nums.end(),
                [](int x) { std::cout << x * x << " "; });
  std::cout << std::endl;

  auto square = [](int x) { return x * x; };

  for (int num : nums) {
    std::cout << square(num) << " ";
  }
  std::cout << std::endl;

  return 0;
}
