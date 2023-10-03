#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

using namespace nb::literals;

using vec3 = nanobind::ndarray<nb::jax, double, nb::shape<3> >;

static double A[3][3] = {{1.0, 2.0, 3.0},
                         {4.0, 5.0, 6.0},
                         {7.0, 8.0, 9.0}};

NB_MODULE(nanobind_example_ext, m) {
  m.def("func", [](vec3 x) { 
    vec3 Ax(new double[3], {3});
    Ax(0) = A[0][0] * x(0) + A[0][1] * x(1) + A[0][2] * x(2);
    Ax(1) = A[1][0] * x(0) + A[1][1] * x(1) + A[1][2] * x(2);
    Ax(2) = A[2][0] * x(0) + A[2][1] * x(1) + A[2][2] * x(2);
    return Ax;
  }, "x"_a);

  m.def("vjp", [](vec3 y) { 
    vec3 yA(new double[3], {3});
    yA(0) = y(0) * A[0][0] + y(1) * A[1][0] + y(2) * A[2][0];
    yA(1) = y(0) * A[0][1] + y(1) * A[1][1] + y(2) * A[2][1];
    yA(2) = y(0) * A[0][2] + y(1) * A[1][2] + y(2) * A[2][2];
    return yA;
  }, "y"_a);
}
