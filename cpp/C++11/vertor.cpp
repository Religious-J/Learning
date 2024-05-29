#include <iostream>
#include <vector>

int main() {
  // 创建一个 vector
  std::vector<int> numbers = {1, 2, 3, 4, 5};

  // 使用 data() 访问 vector 的底层数组
  int *arr_ptr = numbers.data();

  // 打印 vector 中的所有元素
  for (int i = 0; i < numbers.size(); i++) {
    std::cout << arr_ptr[i] << " ";
  }
  std::cout << std::endl;

  // 修改 vector 中的元素
  arr_ptr[2] = 10;

  // 再次打印 vector 中的所有元素
  for (int i = 0; i < numbers.size(); i++) {
    std::cout << numbers[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
