#include "newellasymptotic.hpp"

#include <iostream>
#include <vector>

/** 
 * This function cleans up a vector containing vectors of the form
 * (N,a,b,c,P,f,g,i) by comparing a,b,c,P,f,g and i of one vector with another
 * vector. If those are all equal, the N values are added together and one of
 * them is removed.
 */
std::vector<std::vector<int>> cleanup(std::vector<std::vector<int>> &expansion) {
    std::vector<std::vector<int>> new_expansion;
    int N, a, b, c, P, hx, hy, hz;

    for (size_t i = 0; i < expansion.size(); ++i) {
        auto& term1 = expansion[i];
        N = term1[0];
        a = term1[1], b = term1[2], c = term1[3];
        P = term1[4];
        hx = term1[5], hy = term1[6], hz = term1[7];

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
        expansion[i][0] = N;
    }
    return expansion;
}

/** 
 * The derivatives dx, dy and dz of a function of the form
 * f(x,y,z) = N * hx^f * hy^g * hz^i * x^a * y^b * z^c / R^P 
 * where N is a constant, R = sqrt(x² + y² + z²), hx, hy and hz are cell sizes
 * and x, y and z are coordinates are here calculated.
 * These f(x,y,z) can be rewritten as vectors of the form (N,a,b,c,P,f,g,i).
 * dx, dy and dz accept a vector containing vectors of this form return the
 * derivativesand as a vector containing vectors of that same form

 *  dx: (N,a,b,c,P,f,g,i) --> (-N(P-a),a+1,b,c,P+2,f+1,g,i) +
                              (N*a,a-1,b+2,c,P+2,f+1,g,i) + 
                              (N*a,a-1,b,c+2,P+2,f+1,g,i)
    
 *  dy: (N,a,b,c,P,f,g,i) --> (-N(P-b),a,b+1,c,P+2,f,g+1,i) +
                              (N*b,a+2,b-1,c,P+2,f,g+1,i) + 
                              (N*b,a,b-1,c+2,P+2,f,g+1,i)
    
 *  dz: (N,a,b,c,P,f,g,i) --> (-N(P-c),a,b,c+1,P+2,f,g,i+1) +
                              (N*c,a+2,b,c-1,P+2,f,g,i+1) + 
                              (N*c,a,b+2,c-1,P+2,f,g,i+1)
 */

// dx
const std::vector<std::vector<int>> dxSymCal(std::vector<std::vector<int>> &expansion) {
    std::vector<std::vector<int>> new_expansion;
    int N, a, b, c, P, hx, hy, hz;
    for(const std::vector<int>& term : expansion) {
        // first term
        N = -term[0] * (term[4] - term[1]);
        a = term[1] + 1;
        b = term[2];
        c = term[3];
        P = term[4] + 2;
        hx = term[5] + 1;
        hy = term[6];
        hz = term[7];
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
const std::vector<std::vector<int>> dySymCal(std::vector<std::vector<int>> &expansion) {
    std::vector<std::vector<int>> new_expansion;
    int N, a, b, c, P, hx, hy, hz;
    for(const std::vector<int>& term : expansion) {
        // first term
        N = -term[0] * (term[4] - term[2]);
        a = term[1];
        b = term[2] + 1;
        c = term[3];
        P = term[4] + 2;
        hx = term[5];
        hy = term[6] + 1;
        hz = term[7];
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
const std::vector<std::vector<int>> dzSymCal(std::vector<std::vector<int>> &expansion) {
    std::vector<std::vector<int>> new_expansion;
    int N, a, b, c, P, hx, hy, hz;
    for(const std::vector<int>& term : expansion) {
        // first term
        N = -term[0] * (term[4] - term[3]);
        a = term[1];
        b = term[2];
        c = term[3] + 1;
        P = term[4] + 2;
        hx = term[5];
        hy = term[6];
        hz = term[7] + 1;
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

// Determines the derivatives combinations recursively up to a specified order. Thank you ChatGPT
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
std::vector<std::vector<int>> derivativeCombinations(int max_order) {
    int count = max_order / 2 + 1;
    std::vector<int> even_orders(count);
    for (int i = 0; i < count; ++i)
        even_orders[i] = i * 2;

    std::vector<std::vector<int>> valid_combinations;
    std::vector<int> current(3, 0);

    combinationsRecursive(even_orders, max_order, 0, current, valid_combinations);

    return valid_combinations;
}

/** Determine all terms in the asymptotic expansion. Inputs are an order and a
 *  vector containing vectors of the form (N,a,b,c,P,f,g,i).
 */
std::vector<int> deriveUpToOrder(int order, std::vector<std::vector<int>> expansion) {
    std::vector<std::vector<int>> combos = derivativeCombinations(order);
    int size = expansion.size() * 8;
    std::vector<int> result(size);

    int i = 0;
    for (const std::vector<int>& term : expansion) {
        for (const int val : term) {
            result[i] = val;
            i += 1;
        }
    }

    for (std::vector<int>& der : combos) {
        std::vector<std::vector<int>> new_expansion = expansion;
        for (int i = 0; i < der[0]; i++){  // x derivatives
            new_expansion = dxSymCal(new_expansion);
        }
        for (int i = 0; i < der[1]; i++){  // y derivatives
            new_expansion = dySymCal(new_expansion);
        }
        for (int i = 0; i < der[2]; i++){  // z derivatives
            new_expansion = dzSymCal(new_expansion);
        }

        for (std::vector<int>& term : new_expansion) {
            result.insert(result.end(), term.begin(), term.end());;
        }
    }
    return result;
}