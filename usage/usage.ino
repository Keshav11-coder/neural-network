#include "neural.h"
#include "std/vector.hpp" // std::vector<int> h = {0, 6, 5}; this is possible, as well as vector<int>
#include "std/console.hpp"

void printMatrix(const std::vector<std::vector<float>>& matrix) {
  Serial.print("{\n");
  for (size_t i = 0; i < matrix.size(); i++) {
    Serial.print("  {");
    for (size_t j = 0; j < matrix[i].size(); j++) {
      Serial.print(matrix[i][j], 6);  // Print with 6 decimal places
      if (j < matrix[i].size() - 1) {
        Serial.print(", ");
      }
    }
    Serial.print("}");
    if (i < matrix.size() - 1) {
      Serial.println(",");
    }
  }
  Serial.println("\n}");
}

neural::network_v2 nn2(neural::createWeights_mat(9, 1),
                    new neural::layer(
                      neural::activation::Sigmoid,
                      neural::createWeights_mat(4, 9),
                      neural::createWeights_mat(4, 1)),
                    new neural::layer(
                      neural::activation::Softmax,
                      neural::createWeights_mat(4, 4),
                      neural::createWeights_mat(4, 1)));

void setup() {
  Serial.begin(115200);
}

void loop() {
  std::vector<std::vector<float>> output = neural::feedforward(&nn2);
  Serial.println("Output:");
  printMatrix(output);

  std::vector<std::vector<float>> target = {{0.0}, {0.0}, {0.0}, {1.0}};
  Serial.println("Target:");
  printMatrix(target);

  neural::network_v2 *upd = neural::backpropagate(&nn2, output, target, 0.01); // Reduced learning rate
  delay(1000);
}
