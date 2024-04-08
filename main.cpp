#include <iostream>
#include <cmath>
#include <vector>
#include <functional>

using namespace std;

typedef double (*fn_t)(vector<double> &);

double calculate_partial_derivative(fn_t function, vector<double> &args, int unknown_id, double h = 1e-5) {
    vector<double> args_h = args;
    args_h[unknown_id] += h;
    return (function(args_h) - function(args)) / h;
}

vector<double> calculate_gradient(fn_t function, vector<double> &args, double h = 1e-5) {
    vector<double> gradient;
    for (int i = 0; i < args.size(); i++) {
        gradient.push_back(calculate_partial_derivative(function, args, i, h));
    }
    return gradient;
}

double calculate_abs_grad(vector<double> &grad) {
    double abs = 0.0;
    for (double i: grad) {
        abs += pow(i, 2);
    }
    abs = sqrt(abs);
    return abs;
}

double initial_function(vector<double> &args) {
    return pow(args[0], 3) + pow(args[1], 2) + 2 * pow(args[2], 2) - args[1] * args[2] - args[1];
}

void find_minimum_with_gradient_descent(fn_t function, vector<double> &initial_point, double precision) {
    double step = 1.0;
    double previous_function_value, new_function_value;
    vector<double> gradient;
    vector<double> x = initial_point;
    bool is_convergent;
    do {
        is_convergent = true;
        previous_function_value = function(x);
        gradient = calculate_gradient(function, x);
        for (int i = 0; i < initial_point.size(); i++) {
            x[i] -= step * gradient[i];
        }
        cout << "Новая точка: ";
        for (double coordinate: x) {
            cout << coordinate << "; ";
        }
        cout << endl;
        new_function_value = function(x);
        cout << "Значение функции в ней = " << new_function_value << endl;
        if (new_function_value >= previous_function_value) {
            step /= 2;
            cout << "Это больше, чем в предыдущей точке. Уменьшаем шаг до " << step << endl;
            is_convergent = false;
        }
    } while (calculate_abs_grad(gradient) > precision || !is_convergent);
    cout << "---------" << endl;
    cout << "Решение с точностью " << precision << ": ";
    for (double coordinate: x) {
        cout << coordinate << "; ";
    }
    cout << endl << "Значении функции: " << new_function_value << endl;
}

function<double(double)> find_derivative(const function<double(double)> &function, double h = 1e-5) {
    return [function, h](double x) {
        return (function(x + h) - function(x)) / h;
    };
}

function<double(double)> get_func(fn_t function, const vector<double> &point, const vector<double> &s) {
    return [function, point, s](double h) {
        vector<double> modified_point = point;
        for (int i = 0; i < point.size(); i++) {
            modified_point[i] -= h * s[i];
        }
        return function(modified_point);
    };
}

void find_minimum_with_steepest_descent(fn_t function, vector<double> &initial_point, double precision) {
    vector<double> x = initial_point;
    vector<double> gradient = calculate_gradient(function, x);
    while (calculate_abs_grad(gradient) >= precision) {
        cout << "Модуль градиента больше погрешности. Считаем значение S: ";
        vector<double> s;
        for (int i = 0; i < initial_point.size(); i++) {
            s.push_back(gradient[i] / calculate_abs_grad(gradient));
            cout << gradient[i] / calculate_abs_grad(gradient) << " ";
        }
        cout << endl << "Подставляем в исходную функцию. Ищем h, при котором производная равна нулю: ";
        double h = 0.0, dh = 0.1;
        double derivative = find_derivative(get_func(function, x, s))(h);
        while (abs(derivative) > precision) {
            double derivative_prime = (find_derivative(get_func(function, x, s))(h + dh) - derivative) / dh;
            h -= derivative / derivative_prime;
            derivative = find_derivative(get_func(function, x, s))(h);
        }
        cout << h << endl << "Новая точка: ";
        for (int i = 0; i < x.size(); i++) {
            x[i] -= h * s[i];
            cout << x[i] << " ";
        }
        gradient = calculate_gradient(function, x);
        cout << endl << "Модуль градиента: " << calculate_abs_grad(gradient) << endl;
    }
    cout << "---------" << endl;
    cout << "Решение с точностью " << precision << ": ";
    for (double coordinate: x) {
        cout << coordinate << "; ";
    }
    cout << endl << "Значении функции: " << function(x) << endl;
}

int main() {
    vector<double> initial_point = {0, 0, 0};
    find_minimum_with_gradient_descent(initial_function, initial_point, 0.001);
    find_minimum_with_steepest_descent(initial_function, initial_point, 0.001);
    return 0;
}
