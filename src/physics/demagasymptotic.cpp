#include "demagasymptotic.hpp"

#include <iostream>
#include <vector>

/* Calculate all needed derivatives for the asymptotic expansion and put the
   result in one list.
   The derivatives are always taken of functions of the form
   f = N * dx^d * dy^e * dz^f * x^a * y^b * z^c / R^P
   This results in a function of the same form. Each derivative generates three
   more terms and the terms are shematically represented as tuples of the form
   (N,a,b,c,P,d,e,f)
*/

// dx
std::vector<std::vector<int>> dx(std::vector<std::vector<int>> expansion) {
    std::vector<std::vector<int>> new_expansion;
    for(const std::vector<int>& term : expansion) {
        // first term
        int N = -term[0] * (term[4] - term[1]);
        int a = term[1] + 1;
        int b = term[2];
        int c = term[3];
        int P = term[4] + 2;
        int hx = term[5] + 1;
        int hy = term[6];
        int hz = term[7];
        new_expansion.push_back({N,a,b,c,P,hx,hy,hz});

        if (term[1] != 0) {
            // second term
            N = term[0] * term[1];
            a = term[1] - 1;
            b = term[2] + 2;
            c = term[3];
            P = term[4] + 2;
            hx = term[5] + 1;
            hy = term[6];
            hz = term[7];
            new_expansion.push_back({N,a,b,c,P,hx,hy,hz});

            // third term
            N = term[0] * term[1];
            a = term[1] - 1;
            b = term[2];
            c = term[3] + 2;
            P = term[4] + 2;
            hx = term[5] + 1;
            hy = term[6];
            hz = term[7];
            new_expansion.push_back({N,a,b,c,P,hx,hy,hz});
        }
    }
    return cleanup(new_expansion);
}

// dy
std::vector<std::vector<int>> dy(std::vector<std::vector<int>> expansion) {
    std::vector<std::vector<int>> new_expansion;
    for(const std::vector<int>& term : expansion) {
        // first term
        int N = -term[0] * (term[4] - term[2]);
        int a = term[1];
        int b = term[2] + 1;
        int c = term[3];
        int P = term[4] + 2;
        int hx = term[5];
        int hy = term[6] + 1;
        int hz = term[7];
        new_expansion.push_back({N,a,b,c,P,hx,hy,hz});

        if (term[2] != 0) {
            // second term
            N = term[0] * term[2];
            a = term[1] + 2;
            b = term[2] - 1;
            c = term[3];
            P = term[4] + 2;
            hx = term[5];
            hy = term[6] + 1;
            hz = term[7];
            new_expansion.push_back({N,a,b,c,P,hx,hy,hz});

            // third term
            N = term[0] * term[2];
            a = term[1];
            b = term[2] - 1;
            c = term[3] + 2;
            P = term[4] + 2;
            hx = term[5];
            hy = term[6] + 1;
            hz = term[7];
            new_expansion.push_back({N,a,b,c,P,hx,hy,hz});
        }
    }
    return cleanup(new_expansion);
}

// dz
std::vector<std::vector<int>> dz(std::vector<std::vector<int>> expansion) {
    std::vector<std::vector<int>> new_expansion;
    for(const std::vector<int>& term : expansion) {
        // first term
        int N = -term[0] * (term[4] - term[3]);
        int a = term[1];
        int b = term[2];
        int c = term[3] + 1;
        int P = term[4] + 2;
        int hx = term[5];
        int hy = term[6];
        int hz = term[7] + 1;
        new_expansion.push_back({N,a,b,c,P,hx,hy,hz});

        if (term[3] != 0) {
            // second term
            N = term[0] * term[3];
            a = term[1] + 2;
            b = term[2];
            c = term[3] - 1;
            P = term[4] + 2;
            hx = term[5];
            hy = term[6];
            hz = term[7] + 1;
            new_expansion.push_back({N,a,b,c,P,hx,hy,hz});

            // third term
            N = term[0] * term[3];
            a = term[1];
            b = term[2] + 2;
            c = term[3] - 1;
            P = term[4] + 2;
            hx = term[5];
            hy = term[6];
            hz = term[7] + 1;
            new_expansion.push_back({N,a,b,c,P,hx,hy,hz});
        }
    }
    return cleanup(new_expansion);
}

// Clean up by adding N values of terms with the same qoefficients
std::vector<std::vector<int>> cleanup(std::vector<std::vector<int>> expansion) {
    std::vector<std::vector<int>> new_expansion;
    for (size_t i = 0; i < expansion.size(); ++i) {
        auto& term1 = expansion[i];
        int N = term1[0];
        int a = term1[1], b = term1[2], c = term1[3];
        int P = term1[4];
        int hx = term1[5], hy = term1[6], hz = term1[7];

        for (size_t j = i + 1; j < expansion.size(); ) {
            const auto& term2 = expansion[j];
            if (a == term2[1] && b == term2[2] && c == term2[3] &&
                P == term2[4] && hx == term2[5] && hy == term2[6] && hz == term2[7]) {
                N += term2[0];
                expansion.erase(expansion.begin() + j); // erase shifts elements, don't increment j
            } else {
                ++j;
            }
        }
        new_expansion.push_back({N,a,b,c,P,hx,hy,hz});
    }
    return new_expansion;
}

// Determine combinations, thank you ChatGPT
void combinationsRecursive(
    const std::vector<int>& even_orders,
    int max_order,
    int depth,
    std::vector<int>& current,
    std::vector<std::vector<int>>& result
) {
    if (depth == current.size()) {
        int total = 0;
        for (int v : current) total += v;
        if (total > 0 && total <= max_order)
            result.push_back(current);
        return;
    }

    for (int val : even_orders) {
        current[depth] = val;
        combinationsRecursive(even_orders, max_order, depth + 1, current, result);
    }
}

/* Determine how many times you should take the derivative to x, y and z for a term.
  This returns a vector containing vectors of three values. Those being the
  derivatives to x, y and z.*/
std::vector<std::vector<int>> derivativeCombinations(int num_variables, int max_order) {
    std::vector<int> even_orders;
    for (int i = 0; i <= max_order; i += 2)
        even_orders.push_back(i);

    std::vector<std::vector<int>> valid_combinations;
    std::vector<int> current(num_variables, 0);

    combinationsRecursive(even_orders, max_order, 0, current, valid_combinations);

    return valid_combinations;
}

// Generate all terms up to a specific order (also including that order).
std::vector<int> uptoOrder(int order, std::vector<std::vector<int>> expansion) {
    int num_variables = 3;
    std::vector<std::vector<int>> combos = derivativeCombinations(num_variables, order);
    std::vector<int> result;

    for (const std::vector<int>& term : expansion) {
        for (const int val : term) {
            result.push_back(val);
        }
    }

    for (std::vector<int>& der : combos) {
        std::vector<std::vector<int>> new_expansion = expansion;
        for (int i = 0; i < der[0]; i++){  // x derivatives
            new_expansion = dx(new_expansion);
        }
        for (int i = 0; i < der[1]; i++){  // y derivatives
            new_expansion = dy(new_expansion);
        }
        for (int i = 0; i < der[2]; i++){  // z derivatives
            new_expansion = dz(new_expansion);
        }

        for (std::vector<int>& term : new_expansion) {
            for (const int val : term) {
                result.push_back(val);
            }
        }
    }
    return result;
}