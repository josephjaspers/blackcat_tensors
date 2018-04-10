
Author Joseph Jaspers

--More documentation that needs to be added to primary README.

- operator = (Tensor_Type&& move_) swaps the two internal types, ignore dimension checking (well defined)
- operator = (Tensor_Type&  copy_) DOES check for matching dimensions

