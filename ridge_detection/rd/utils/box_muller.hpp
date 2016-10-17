/* The authors of this work have released all rights to it and placed it
	in the public domain under the Creative Commons CC0 1.0 waiver
	(http://creativecommons.org/publicdomain/zero/1.0/).

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
	EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
	MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
	IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
	CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
	TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
	SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	Retrieved from: http://en.literateprograms.org/Box-Muller_transform_(C)?oldid=7011
*/
/**
 * @file box_muller.hpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering, Faculty of Electronics and Information
 * Technology, Warsaw University of Technology 2016
 *
 *
 * @note Modified to use templates
 */

#ifndef BOX_MULLER_HPP_
#define BOX_MULLER_HPP_

#include <stdlib.h>
#include <cmath>

namespace rd
{

template <typename T>
T rand_normal(T mean, T stddev) {
    static T n2 = 0.0;
    static int n2_cached = 0;

    if (!n2_cached) {
        T x, y, r;
		do {
			x = 2.0*rand()/RAND_MAX - 1;
			y = 2.0*rand()/RAND_MAX - 1;

			r = x*x + y*y;
		} while (r == 0.0 || r > 1.0);

		T d = std::sqrt(-2.0*std::log(r)/r);
		T n1 = x*d;
		n2 = y*d;
		T result = n1*stddev + mean;
		n2_cached = 1;
		return result;
    } else {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}

} // end namespace rd

#endif // BOX_MULLER_HPP_
