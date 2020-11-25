import ccl

print(1)
bool_image = [[1, 1, 0, 0],
              [-0, 0, 0, 0],
              [-0, 0, 1, 1]]
print(2)
result = ccl.connected_component_labelling(bool_image, 4)
print(result)
