#include "newellasymptotic.hpp"


/** Calculate the asymptotic solution of the demagkernel component Nxx. 
    This method is based on the method used in OOMMF.
    https://math.nist.gov/~MDonahue/talks/mmm2020-talk.pdf
 */
__host__ __device__ real calcAsymptoticNxx(int3 idx, real3 cellsize, int order) {

  double hx = cellsize.x;
  double hy = cellsize.y;
  double hz = cellsize.z;
  double x = idx.x * hx;
  double y = idx.y * hy;
  double z = idx.z * hz;
  double R = sqrt(x*x + y*y + z*z);
  
  double result = 0;

  if (!(order % 2)) {order -= 1;}

  switch(order) {
    case 11:
      result += (70761600./14515200. * x*x*pow(z,8)*pow(hz,8)
              + -282592800./14515200. * pow(x,4)*pow(z,6)*pow(hz,8)
              + -285768000./14515200. * x*x*y*y*pow(z,6)*pow(hz,8)
              + 250047000./14515200. * pow(x,6)*pow(z,4)*pow(hz,8)
              + 488187000./14515200. * pow(x,4)*y*y*pow(z,4)*pow(hz,8)
              + 226233000./14515200. * x*x*pow(y,4)*pow(z,4)*pow(hz,8)
              + -48521025./14515200. * pow(x,8)*z*z*pow(hz,8)
              + -141693300./14515200. * pow(x,6)*y*y*z*z*pow(hz,8)
              + -133953750./14515200. * pow(x,4)*pow(y,4)*z*z*pow(hz,8)
              + -36911700./14515200. * x*x*pow(y,6)*z*z*pow(hz,8)
              + 992250./14515200. * pow(x,10)*pow(hz,8)
              + 3869775./14515200. * pow(x,8)*y*y*pow(hz,8)
              + 5556600./14515200. * pow(x,6)*pow(y,4)*pow(hz,8)
              + 3373650./14515200. * pow(x,4)*pow(y,6)*pow(hz,8)
              + 595350./14515200. * x*x*pow(y,8)*pow(hz,8)
              + 10886400./14515200. * y*y*pow(z,8)*pow(hz,8)
              + -3175200./14515200. * pow(y,4)*pow(z,6)*pow(hz,8)
              + -11907000./14515200. * pow(y,6)*pow(z,4)*pow(hz,8)
              + 3869775./14515200. * pow(y,8)*z*z*pow(hz,8)
              + -99225./14515200. * pow(y,10)*pow(hz,8)
              + -1814400./14515200. * pow(z,10)*pow(hz,8)
              + 241712100./1935360. * x*x*y*y*pow(z,6)*hy*hy*pow(hz,6)
              + -302140125./1935360. * pow(x,4)*y*y*pow(z,4)*hy*hy*pow(hz,6)
              + -302140125./1935360. * x*x*pow(y,4)*pow(z,4)*hy*hy*pow(hz,6)
              + 64893150./1935360. * pow(x,6)*y*y*z*z*hy*hy*pow(hz,6)
              + 139907250./1935360. * pow(x,4)*pow(y,4)*z*z*hy*hy*pow(hz,6)
              + 64893150./1935360. * x*x*pow(y,6)*z*z*hy*hy*pow(hz,6)
              + -1318275./1935360. * pow(x,8)*y*y*hy*hy*pow(hz,6)
              + -4663575./1935360. * pow(x,6)*pow(y,4)*hy*hy*pow(hz,6)
              + -4663575./1935360. * pow(x,4)*pow(y,6)*hy*hy*pow(hz,6)
              + -1318275./1935360. * x*x*pow(y,8)*hy*hy*pow(hz,6)
              + 7342650./1935360. * pow(x,4)*pow(z,6)*hy*hy*pow(hz,6)
              + 12800025./1935360. * pow(x,6)*pow(z,4)*hy*hy*pow(hz,6)
              + -5060475./1935360. * pow(x,8)*z*z*hy*hy*pow(hz,6)
              + 141750./1935360. * pow(x,10)*hy*hy*pow(hz,6)
              + -10206000./1935360. * x*x*pow(z,8)*hy*hy*pow(hz,6)
              + 7342650./1935360. * pow(y,4)*pow(z,6)*hy*hy*pow(hz,6)
              + 12800025./1935360. * pow(y,6)*pow(z,4)*hy*hy*pow(hz,6)
              + -5060475./1935360. * pow(y,8)*z*z*hy*hy*pow(hz,6)
              + 141750./1935360. * pow(y,10)*hy*hy*pow(hz,6)
              + -10206000./1935360. * y*y*pow(z,8)*hy*hy*pow(hz,6)
              + 453600./1935360. * pow(z,10)*hy*hy*pow(hz,6)
              + 372093750./1036800. * x*x*pow(y,4)*pow(z,4)*pow(hy,4)*pow(hz,4)
              + -69953625./1036800. * pow(x,4)*pow(y,4)*z*z*pow(hy,4)*pow(hz,4)
              + -120856050./1036800. * x*x*pow(y,6)*z*z*pow(hy,4)*pow(hz,4)
              + -396900./1036800. * pow(x,6)*pow(y,4)*pow(hy,4)*pow(hz,4)
              + 5060475./1036800. * pow(x,4)*pow(y,6)*pow(hy,4)*pow(hz,4)
              + 3231900./1036800. * x*x*pow(y,8)*pow(hy,4)*pow(hz,4)
              + -69953625./1036800. * pow(x,4)*y*y*pow(z,4)*pow(hy,4)*pow(hz,4)
              + 55962900./1036800. * pow(x,6)*y*y*z*z*pow(hy,4)*pow(hz,4)
              + -1913625./1036800. * pow(x,8)*y*y*pow(hy,4)*pow(hz,4)
              + -120856050./1036800. * x*x*y*y*pow(z,6)*pow(hy,4)*pow(hz,4)
              + -396900./1036800. * pow(x,6)*pow(z,4)*pow(hy,4)*pow(hz,4)
              + -1913625./1036800. * pow(x,8)*z*z*pow(hy,4)*pow(hz,4)
              + 85050./1036800. * pow(x,10)*pow(hy,4)*pow(hz,4)
              + 5060475./1036800. * pow(x,4)*pow(z,6)*pow(hy,4)*pow(hz,4)
              + 3231900./1036800. * x*x*pow(z,8)*pow(hy,4)*pow(hz,4)
              + -12403125./1036800. * pow(y,6)*pow(z,4)*pow(hy,4)*pow(hz,4)
              + 6974100./1036800. * pow(y,8)*z*z*pow(hy,4)*pow(hz,4)
              + -226800./1036800. * pow(y,10)*pow(hy,4)*pow(hz,4)
              + -12403125./1036800. * pow(y,4)*pow(z,6)*pow(hy,4)*pow(hz,4)
              + 6974100./1036800. * y*y*pow(z,8)*pow(hy,4)*pow(hz,4)
              + -226800./1036800. * pow(z,10)*pow(hy,4)*pow(hz,4)
              + 241712100./1935360. * x*x*pow(y,6)*z*z*pow(hy,6)*hz*hz
              + 7342650./1935360. * pow(x,4)*pow(y,6)*pow(hy,6)*hz*hz
              + -10206000./1935360. * x*x*pow(y,8)*pow(hy,6)*hz*hz
              + -302140125./1935360. * pow(x,4)*pow(y,4)*z*z*pow(hy,6)*hz*hz
              + 12800025./1935360. * pow(x,6)*pow(y,4)*pow(hy,6)*hz*hz
              + -302140125./1935360. * x*x*pow(y,4)*pow(z,4)*pow(hy,6)*hz*hz
              + 64893150./1935360. * pow(x,6)*y*y*z*z*pow(hy,6)*hz*hz
              + -5060475./1935360. * pow(x,8)*y*y*pow(hy,6)*hz*hz
              + 139907250./1935360. * pow(x,4)*y*y*pow(z,4)*pow(hy,6)*hz*hz
              + 64893150./1935360. * x*x*y*y*pow(z,6)*pow(hy,6)*hz*hz
              + -1318275./1935360. * pow(x,8)*z*z*pow(hy,6)*hz*hz
              + 141750./1935360. * pow(x,10)*pow(hy,6)*hz*hz
              + -4663575./1935360. * pow(x,6)*pow(z,4)*pow(hy,6)*hz*hz
              + -4663575./1935360. * pow(x,4)*pow(z,6)*pow(hy,6)*hz*hz
              + -1318275./1935360. * x*x*pow(z,8)*pow(hy,6)*hz*hz
              + -10206000./1935360. * pow(y,8)*z*z*pow(hy,6)*hz*hz
              + 453600./1935360. * pow(y,10)*pow(hy,6)*hz*hz
              + 7342650./1935360. * pow(y,6)*pow(z,4)*pow(hy,6)*hz*hz
              + 12800025./1935360. * pow(y,4)*pow(z,6)*pow(hy,6)*hz*hz
              + -5060475./1935360. * y*y*pow(z,8)*pow(hy,6)*hz*hz
              + 141750./1935360. * pow(z,10)*pow(hy,6)*hz*hz
              + 70761600./14515200. * x*x*pow(y,8)*pow(hy,8)
              + -282592800./14515200. * pow(x,4)*pow(y,6)*pow(hy,8)
              + -285768000./14515200. * x*x*pow(y,6)*z*z*pow(hy,8)
              + 250047000./14515200. * pow(x,6)*pow(y,4)*pow(hy,8)
              + 488187000./14515200. * pow(x,4)*pow(y,4)*z*z*pow(hy,8)
              + 226233000./14515200. * x*x*pow(y,4)*pow(z,4)*pow(hy,8)
              + -48521025./14515200. * pow(x,8)*y*y*pow(hy,8)
              + -141693300./14515200. * pow(x,6)*y*y*z*z*pow(hy,8)
              + -133953750./14515200. * pow(x,4)*y*y*pow(z,4)*pow(hy,8)
              + -36911700./14515200. * x*x*y*y*pow(z,6)*pow(hy,8)
              + 992250./14515200. * pow(x,10)*pow(hy,8)
              + 3869775./14515200. * pow(x,8)*z*z*pow(hy,8)
              + 5556600./14515200. * pow(x,6)*pow(z,4)*pow(hy,8)
              + 3373650./14515200. * pow(x,4)*pow(z,6)*pow(hy,8)
              + 595350./14515200. * x*x*pow(z,8)*pow(hy,8)
              + -1814400./14515200. * pow(y,10)*pow(hy,8)
              + 10886400./14515200. * pow(y,8)*z*z*pow(hy,8)
              + -3175200./14515200. * pow(y,6)*pow(z,4)*pow(hy,8)
              + -11907000./14515200. * pow(y,4)*pow(z,6)*pow(hy,8)
              + 3869775./14515200. * y*y*pow(z,8)*pow(hy,8)
              + -99225./14515200. * pow(z,10)*pow(hy,8)
              + 275250150./1935360. * pow(x,4)*pow(z,6)*hx*hx*pow(hz,6)
              + -262847025./1935360. * pow(x,6)*pow(z,4)*hx*hx*pow(hz,6)
              + -186046875./1935360. * pow(x,4)*y*y*pow(z,4)*hx*hx*pow(hz,6)
              + 53581500./1935360. * pow(x,8)*z*z*hx*hx*pow(hz,6)
              + 76800150./1935360. * pow(x,6)*y*y*z*z*hx*hx*pow(hz,6)
              + -5953500./1935360. * pow(x,4)*pow(y,4)*z*z*hx*hx*pow(hz,6)
              + -1134000./1935360. * pow(x,10)*hx*hx*pow(hz,6)
              + -2551500./1935360. * pow(x,8)*y*y*hx*hx*pow(hz,6)
              + -893025./1935360. * pow(x,6)*pow(y,4)*hx*hx*pow(hz,6)
              + 1289925./1935360. * pow(x,4)*pow(y,6)*hx*hx*pow(hz,6)
              + 44055900./1935360. * x*x*y*y*pow(z,6)*hx*hx*pow(hz,6)
              + 75907125./1935360. * x*x*pow(y,4)*pow(z,4)*hx*hx*pow(hz,6)
              + -27981450./1935360. * x*x*pow(y,6)*z*z*hx*hx*pow(hz,6)
              + 722925./1935360. * x*x*pow(y,8)*hx*hx*pow(hz,6)
              + -60555600./1935360. * x*x*pow(z,8)*hx*hx*pow(hz,6)
              + -4167450./1935360. * pow(y,4)*pow(z,6)*hx*hx*pow(hz,6)
              + -893025./1935360. * pow(y,6)*pow(z,4)*hx*hx*pow(hz,6)
              + 1190700./1935360. * pow(y,8)*z*z*hx*hx*pow(hz,6)
              + -42525./1935360. * pow(y,10)*hx*hx*pow(hz,6)
              + -680400./1935360. * y*y*pow(z,8)*hx*hx*pow(hz,6)
              + 1360800./1935360. * pow(z,10)*hx*hx*pow(hz,6)
              + 372093750./414720. * pow(x,4)*y*y*pow(z,4)*hx*hx*hy*hy*pow(hz,4)
              + -120856050./414720. * pow(x,6)*y*y*z*z*hx*hx*hy*hy*pow(hz,4)
              + -69953625./414720. * pow(x,4)*pow(y,4)*z*z*hx*hx*hy*hy*pow(hz,4)
              + 3231900./414720. * pow(x,8)*y*y*hx*hx*hy*hy*pow(hz,4)
              + 5060475./414720. * pow(x,6)*pow(y,4)*hx*hx*hy*hy*pow(hz,4)
              + -396900./414720. * pow(x,4)*pow(y,6)*hx*hx*hy*hy*pow(hz,4)
              + -12403125./414720. * pow(x,6)*pow(z,4)*hx*hx*hy*hy*pow(hz,4)
              + 6974100./414720. * pow(x,8)*z*z*hx*hx*hy*hy*pow(hz,4)
              + -226800./414720. * pow(x,10)*hx*hx*hy*hy*pow(hz,4)
              + -12403125./414720. * pow(x,4)*pow(z,6)*hx*hx*hy*hy*pow(hz,4)
              + -69953625./414720. * x*x*pow(y,4)*pow(z,4)*hx*hx*hy*hy*pow(hz,4)
              + 55962900./414720. * x*x*pow(y,6)*z*z*hx*hx*hy*hy*pow(hz,4)
              + -1913625./414720. * x*x*pow(y,8)*hx*hx*hy*hy*pow(hz,4)
              + -120856050./414720. * x*x*y*y*pow(z,6)*hx*hx*hy*hy*pow(hz,4)
              + 6974100./414720. * x*x*pow(z,8)*hx*hx*hy*hy*pow(hz,4)
              + -396900./414720. * pow(y,6)*pow(z,4)*hx*hx*hy*hy*pow(hz,4)
              + -1913625./414720. * pow(y,8)*z*z*hx*hx*hy*hy*pow(hz,4)
              + 85050./414720. * pow(y,10)*hx*hx*hy*hy*pow(hz,4)
              + 5060475./414720. * pow(y,4)*pow(z,6)*hx*hx*hy*hy*pow(hz,4)
              + 3231900./414720. * y*y*pow(z,8)*hx*hx*hy*hy*pow(hz,4)
              + -226800./414720. * pow(z,10)*hx*hx*hy*hy*pow(hz,4)
              + 372093750./414720. * pow(x,4)*pow(y,4)*z*z*hx*hx*pow(hy,4)*hz*hz
              + -12403125./414720. * pow(x,6)*pow(y,4)*hx*hx*pow(hy,4)*hz*hz
              + -12403125./414720. * pow(x,4)*pow(y,6)*hx*hx*pow(hy,4)*hz*hz
              + -120856050./414720. * pow(x,6)*y*y*z*z*hx*hx*pow(hy,4)*hz*hz
              + 6974100./414720. * pow(x,8)*y*y*hx*hx*pow(hy,4)*hz*hz
              + -69953625./414720. * pow(x,4)*y*y*pow(z,4)*hx*hx*pow(hy,4)*hz*hz
              + 3231900./414720. * pow(x,8)*z*z*hx*hx*pow(hy,4)*hz*hz
              + -226800./414720. * pow(x,10)*hx*hx*pow(hy,4)*hz*hz
              + 5060475./414720. * pow(x,6)*pow(z,4)*hx*hx*pow(hy,4)*hz*hz
              + -396900./414720. * pow(x,4)*pow(z,6)*hx*hx*pow(hy,4)*hz*hz
              + -120856050./414720. * x*x*pow(y,6)*z*z*hx*hx*pow(hy,4)*hz*hz
              + 6974100./414720. * x*x*pow(y,8)*hx*hx*pow(hy,4)*hz*hz
              + -69953625./414720. * x*x*pow(y,4)*pow(z,4)*hx*hx*pow(hy,4)*hz*hz
              + 55962900./414720. * x*x*y*y*pow(z,6)*hx*hx*pow(hy,4)*hz*hz
              + -1913625./414720. * x*x*pow(z,8)*hx*hx*pow(hy,4)*hz*hz
              + 3231900./414720. * pow(y,8)*z*z*hx*hx*pow(hy,4)*hz*hz
              + -226800./414720. * pow(y,10)*hx*hx*pow(hy,4)*hz*hz
              + 5060475./414720. * pow(y,6)*pow(z,4)*hx*hx*pow(hy,4)*hz*hz
              + -396900./414720. * pow(y,4)*pow(z,6)*hx*hx*pow(hy,4)*hz*hz
              + -1913625./414720. * y*y*pow(z,8)*hx*hx*pow(hy,4)*hz*hz
              + 85050./414720. * pow(z,10)*hx*hx*pow(hy,4)*hz*hz
              + 275250150./1935360. * pow(x,4)*pow(y,6)*hx*hx*pow(hy,6)
              + -262847025./1935360. * pow(x,6)*pow(y,4)*hx*hx*pow(hy,6)
              + -186046875./1935360. * pow(x,4)*pow(y,4)*z*z*hx*hx*pow(hy,6)
              + 53581500./1935360. * pow(x,8)*y*y*hx*hx*pow(hy,6)
              + 76800150./1935360. * pow(x,6)*y*y*z*z*hx*hx*pow(hy,6)
              + -5953500./1935360. * pow(x,4)*y*y*pow(z,4)*hx*hx*pow(hy,6)
              + -1134000./1935360. * pow(x,10)*hx*hx*pow(hy,6)
              + -2551500./1935360. * pow(x,8)*z*z*hx*hx*pow(hy,6)
              + -893025./1935360. * pow(x,6)*pow(z,4)*hx*hx*pow(hy,6)
              + 1289925./1935360. * pow(x,4)*pow(z,6)*hx*hx*pow(hy,6)
              + -60555600./1935360. * x*x*pow(y,8)*hx*hx*pow(hy,6)
              + 44055900./1935360. * x*x*pow(y,6)*z*z*hx*hx*pow(hy,6)
              + 75907125./1935360. * x*x*pow(y,4)*pow(z,4)*hx*hx*pow(hy,6)
              + -27981450./1935360. * x*x*y*y*pow(z,6)*hx*hx*pow(hy,6)
              + 722925./1935360. * x*x*pow(z,8)*hx*hx*pow(hy,6)
              + 1360800./1935360. * pow(y,10)*hx*hx*pow(hy,6)
              + -680400./1935360. * pow(y,8)*z*z*hx*hx*pow(hy,6)
              + -4167450./1935360. * pow(y,6)*pow(z,4)*hx*hx*pow(hy,6)
              + -893025./1935360. * pow(y,4)*pow(z,6)*hx*hx*pow(hy,6)
              + 1190700./1935360. * y*y*pow(z,8)*hx*hx*pow(hy,6)
              + -42525./1935360. * pow(z,10)*hx*hx*pow(hy,6)
              + 275250150./1036800. * pow(x,6)*pow(z,4)*pow(hx,4)*pow(hz,4)
              + -60555600./1036800. * pow(x,8)*z*z*pow(hx,4)*pow(hz,4)
              + 44055900./1036800. * pow(x,6)*y*y*z*z*pow(hx,4)*pow(hz,4)
              + 1360800./1036800. * pow(x,10)*pow(hx,4)*pow(hz,4)
              + -680400./1036800. * pow(x,8)*y*y*pow(hx,4)*pow(hz,4)
              + -4167450./1036800. * pow(x,6)*pow(y,4)*pow(hx,4)*pow(hz,4)
              + -186046875./1036800. * pow(x,4)*y*y*pow(z,4)*pow(hx,4)*pow(hz,4)
              + 75907125./1036800. * pow(x,4)*pow(y,4)*z*z*pow(hx,4)*pow(hz,4)
              + -893025./1036800. * pow(x,4)*pow(y,6)*pow(hx,4)*pow(hz,4)
              + -262847025./1036800. * pow(x,4)*pow(z,6)*pow(hx,4)*pow(hz,4)
              + -5953500./1036800. * x*x*pow(y,4)*pow(z,4)*pow(hx,4)*pow(hz,4)
              + -27981450./1036800. * x*x*pow(y,6)*z*z*pow(hx,4)*pow(hz,4)
              + 1190700./1036800. * x*x*pow(y,8)*pow(hx,4)*pow(hz,4)
              + 76800150./1036800. * x*x*y*y*pow(z,6)*pow(hx,4)*pow(hz,4)
              + 53581500./1036800. * x*x*pow(z,8)*pow(hx,4)*pow(hz,4)
              + 1289925./1036800. * pow(y,6)*pow(z,4)*pow(hx,4)*pow(hz,4)
              + 722925./1036800. * pow(y,8)*z*z*pow(hx,4)*pow(hz,4)
              + -42525./1036800. * pow(y,10)*pow(hx,4)*pow(hz,4)
              + -893025./1036800. * pow(y,4)*pow(z,6)*pow(hx,4)*pow(hz,4)
              + -2551500./1036800. * y*y*pow(z,8)*pow(hx,4)*pow(hz,4)
              + -1134000./1036800. * pow(z,10)*pow(hx,4)*pow(hz,4)
              + 241712100./414720. * pow(x,6)*y*y*z*z*pow(hx,4)*hy*hy*hz*hz
              + -10206000./414720. * pow(x,8)*y*y*pow(hx,4)*hy*hy*hz*hz
              + 7342650./414720. * pow(x,6)*pow(y,4)*pow(hx,4)*hy*hy*hz*hz
              + -10206000./414720. * pow(x,8)*z*z*pow(hx,4)*hy*hy*hz*hz
              + 453600./414720. * pow(x,10)*pow(hx,4)*hy*hy*hz*hz
              + 7342650./414720. * pow(x,6)*pow(z,4)*pow(hx,4)*hy*hy*hz*hz
              + -302140125./414720. * pow(x,4)*pow(y,4)*z*z*pow(hx,4)*hy*hy*hz*hz
              + 12800025./414720. * pow(x,4)*pow(y,6)*pow(hx,4)*hy*hy*hz*hz
              + -302140125./414720. * pow(x,4)*y*y*pow(z,4)*pow(hx,4)*hy*hy*hz*hz
              + 12800025./414720. * pow(x,4)*pow(z,6)*pow(hx,4)*hy*hy*hz*hz
              + 64893150./414720. * x*x*pow(y,6)*z*z*pow(hx,4)*hy*hy*hz*hz
              + -5060475./414720. * x*x*pow(y,8)*pow(hx,4)*hy*hy*hz*hz
              + 139907250./414720. * x*x*pow(y,4)*pow(z,4)*pow(hx,4)*hy*hy*hz*hz
              + 64893150./414720. * x*x*y*y*pow(z,6)*pow(hx,4)*hy*hy*hz*hz
              + -5060475./414720. * x*x*pow(z,8)*pow(hx,4)*hy*hy*hz*hz
              + -1318275./414720. * pow(y,8)*z*z*pow(hx,4)*hy*hy*hz*hz
              + 141750./414720. * pow(y,10)*pow(hx,4)*hy*hy*hz*hz
              + -4663575./414720. * pow(y,6)*pow(z,4)*pow(hx,4)*hy*hy*hz*hz
              + -4663575./414720. * pow(y,4)*pow(z,6)*pow(hx,4)*hy*hy*hz*hz
              + -1318275./414720. * y*y*pow(z,8)*pow(hx,4)*hy*hy*hz*hz
              + 141750./414720. * pow(z,10)*pow(hx,4)*hy*hy*hz*hz
              + 275250150./1036800. * pow(x,6)*pow(y,4)*pow(hx,4)*pow(hy,4)
              + -60555600./1036800. * pow(x,8)*y*y*pow(hx,4)*pow(hy,4)
              + 44055900./1036800. * pow(x,6)*y*y*z*z*pow(hx,4)*pow(hy,4)
              + 1360800./1036800. * pow(x,10)*pow(hx,4)*pow(hy,4)
              + -680400./1036800. * pow(x,8)*z*z*pow(hx,4)*pow(hy,4)
              + -4167450./1036800. * pow(x,6)*pow(z,4)*pow(hx,4)*pow(hy,4)
              + -262847025./1036800. * pow(x,4)*pow(y,6)*pow(hx,4)*pow(hy,4)
              + -186046875./1036800. * pow(x,4)*pow(y,4)*z*z*pow(hx,4)*pow(hy,4)
              + 75907125./1036800. * pow(x,4)*y*y*pow(z,4)*pow(hx,4)*pow(hy,4)
              + -893025./1036800. * pow(x,4)*pow(z,6)*pow(hx,4)*pow(hy,4)
              + 53581500./1036800. * x*x*pow(y,8)*pow(hx,4)*pow(hy,4)
              + 76800150./1036800. * x*x*pow(y,6)*z*z*pow(hx,4)*pow(hy,4)
              + -5953500./1036800. * x*x*pow(y,4)*pow(z,4)*pow(hx,4)*pow(hy,4)
              + -27981450./1036800. * x*x*y*y*pow(z,6)*pow(hx,4)*pow(hy,4)
              + 1190700./1036800. * x*x*pow(z,8)*pow(hx,4)*pow(hy,4)
              + -1134000./1036800. * pow(y,10)*pow(hx,4)*pow(hy,4)
              + -2551500./1036800. * pow(y,8)*z*z*pow(hx,4)*pow(hy,4)
              + -893025./1036800. * pow(y,6)*pow(z,4)*pow(hx,4)*pow(hy,4)
              + 1289925./1036800. * pow(y,4)*pow(z,6)*pow(hx,4)*pow(hy,4)
              + 722925./1036800. * y*y*pow(z,8)*pow(hx,4)*pow(hy,4)
              + -42525./1036800. * pow(z,10)*pow(hx,4)*pow(hy,4)
              + 70761600./1935360. * pow(x,8)*z*z*pow(hx,6)*hz*hz
              + -1814400./1935360. * pow(x,10)*pow(hx,6)*hz*hz
              + 10886400./1935360. * pow(x,8)*y*y*pow(hx,6)*hz*hz
              + -285768000./1935360. * pow(x,6)*y*y*z*z*pow(hx,6)*hz*hz
              + -3175200./1935360. * pow(x,6)*pow(y,4)*pow(hx,6)*hz*hz
              + -282592800./1935360. * pow(x,6)*pow(z,4)*pow(hx,6)*hz*hz
              + 226233000./1935360. * pow(x,4)*pow(y,4)*z*z*pow(hx,6)*hz*hz
              + -11907000./1935360. * pow(x,4)*pow(y,6)*pow(hx,6)*hz*hz
              + 488187000./1935360. * pow(x,4)*y*y*pow(z,4)*pow(hx,6)*hz*hz
              + 250047000./1935360. * pow(x,4)*pow(z,6)*pow(hx,6)*hz*hz
              + -36911700./1935360. * x*x*pow(y,6)*z*z*pow(hx,6)*hz*hz
              + 3869775./1935360. * x*x*pow(y,8)*pow(hx,6)*hz*hz
              + -133953750./1935360. * x*x*pow(y,4)*pow(z,4)*pow(hx,6)*hz*hz
              + -141693300./1935360. * x*x*y*y*pow(z,6)*pow(hx,6)*hz*hz
              + -48521025./1935360. * x*x*pow(z,8)*pow(hx,6)*hz*hz
              + 595350./1935360. * pow(y,8)*z*z*pow(hx,6)*hz*hz
              + -99225./1935360. * pow(y,10)*pow(hx,6)*hz*hz
              + 3373650./1935360. * pow(y,6)*pow(z,4)*pow(hx,6)*hz*hz
              + 5556600./1935360. * pow(y,4)*pow(z,6)*pow(hx,6)*hz*hz
              + 3869775./1935360. * y*y*pow(z,8)*pow(hx,6)*hz*hz
              + 992250./1935360. * pow(z,10)*pow(hx,6)*hz*hz
              + 70761600./1935360. * pow(x,8)*y*y*pow(hx,6)*hy*hy
              + -1814400./1935360. * pow(x,10)*pow(hx,6)*hy*hy
              + 10886400./1935360. * pow(x,8)*z*z*pow(hx,6)*hy*hy
              + -282592800./1935360. * pow(x,6)*pow(y,4)*pow(hx,6)*hy*hy
              + -285768000./1935360. * pow(x,6)*y*y*z*z*pow(hx,6)*hy*hy
              + -3175200./1935360. * pow(x,6)*pow(z,4)*pow(hx,6)*hy*hy
              + 250047000./1935360. * pow(x,4)*pow(y,6)*pow(hx,6)*hy*hy
              + 488187000./1935360. * pow(x,4)*pow(y,4)*z*z*pow(hx,6)*hy*hy
              + 226233000./1935360. * pow(x,4)*y*y*pow(z,4)*pow(hx,6)*hy*hy
              + -11907000./1935360. * pow(x,4)*pow(z,6)*pow(hx,6)*hy*hy
              + -48521025./1935360. * x*x*pow(y,8)*pow(hx,6)*hy*hy
              + -141693300./1935360. * x*x*pow(y,6)*z*z*pow(hx,6)*hy*hy
              + -133953750./1935360. * x*x*pow(y,4)*pow(z,4)*pow(hx,6)*hy*hy
              + -36911700./1935360. * x*x*y*y*pow(z,6)*pow(hx,6)*hy*hy
              + 3869775./1935360. * x*x*pow(z,8)*pow(hx,6)*hy*hy
              + 992250./1935360. * pow(y,10)*pow(hx,6)*hy*hy
              + 3869775./1935360. * pow(y,8)*z*z*pow(hx,6)*hy*hy
              + 5556600./1935360. * pow(y,6)*pow(z,4)*pow(hx,6)*hy*hy
              + 3373650./1935360. * pow(y,4)*pow(z,6)*pow(hx,6)*hy*hy
              + 595350./1935360. * y*y*pow(z,8)*pow(hx,6)*hy*hy
              + -99225./1935360. * pow(z,10)*pow(hx,6)*hy*hy
              + 3628800./14515200. * pow(x,10)*pow(hx,8)
              + -81648000./14515200. * pow(x,8)*y*y*pow(hx,8)
              + -81648000./14515200. * pow(x,8)*z*z*pow(hx,8)
              + 285768000./14515200. * pow(x,6)*pow(y,4)*pow(hx,8)
              + 571536000./14515200. * pow(x,6)*y*y*z*z*pow(hx,8)
              + 285768000./14515200. * pow(x,6)*pow(z,4)*pow(hx,8)
              + -238140000./14515200. * pow(x,4)*pow(y,6)*pow(hx,8)
              + -714420000./14515200. * pow(x,4)*pow(y,4)*z*z*pow(hx,8)
              + -714420000./14515200. * pow(x,4)*y*y*pow(z,4)*pow(hx,8)
              + -238140000./14515200. * pow(x,4)*pow(z,6)*pow(hx,8)
              + 44651250./14515200. * x*x*pow(y,8)*pow(hx,8)
              + 178605000./14515200. * x*x*pow(y,6)*z*z*pow(hx,8)
              + 267907500./14515200. * x*x*pow(y,4)*pow(z,4)*pow(hx,8)
              + 178605000./14515200. * x*x*y*y*pow(z,6)*pow(hx,8)
              + 44651250./14515200. * x*x*pow(z,8)*pow(hx,8)
              + -893025./14515200. * pow(y,10)*pow(hx,8)
              + -4465125./14515200. * pow(y,8)*z*z*pow(hx,8)
              + -8930250./14515200. * pow(y,6)*pow(z,4)*pow(hx,8)
              + -8930250./14515200. * pow(y,4)*pow(z,6)*pow(hx,8)
              + -4465125./14515200. * y*y*pow(z,8)*pow(hx,8)
              + -893025./14515200. * pow(z,10)*pow(hx,8))
              / pow(R,21);

    case 9: 
      result += (509040./161280. * x*x*pow(z,6)*pow(hz,6)
              + -1096200./161280. * pow(x,4)*pow(z,4)*pow(hz,6)
              + -1058400./161280. * x*x*y*y*pow(z,4)*pow(hz,6)
              + 389025./161280. * pow(x,6)*z*z*pow(hz,6)
              + 741825./161280. * pow(x,4)*y*y*z*z*pow(hz,6)
              + 316575./161280. * x*x*pow(y,4)*z*z*pow(hz,6)
              + -12600./161280. * pow(x,8)*pow(hz,6)
              + -36225./161280. * pow(x,6)*y*y*pow(hz,6)
              + -33075./161280. * pow(x,4)*pow(y,4)*pow(hz,6)
              + -7875./161280. * x*x*pow(y,6)*pow(hz,6)
              + 55440./161280. * y*y*pow(z,6)*pow(hz,6)
              + 37800./161280. * pow(y,4)*pow(z,4)*pow(hz,6)
              + -36225./161280. * pow(y,6)*z*z*pow(hz,6)
              + 1575./161280. * pow(y,8)*pow(hz,6)
              + -20160./161280. * pow(z,8)*pow(hz,6)
              + 1200150./34560. * x*x*y*y*pow(z,4)*hy*hy*pow(hz,4)
              + -600075./34560. * pow(x,4)*y*y*z*z*hy*hy*pow(hz,4)
              + -600075./34560. * x*x*pow(y,4)*z*z*hy*hy*pow(hz,4)
              + 21105./34560. * pow(x,6)*y*y*hy*hy*pow(hz,4)
              + 47250./34560. * pow(x,4)*pow(y,4)*hy*hy*pow(hz,4)
              + 21105./34560. * x*x*pow(y,6)*hy*hy*pow(hz,4)
              + -23625./34560. * pow(x,4)*pow(z,4)*hy*hy*pow(hz,4)
              + 49455./34560. * pow(x,6)*z*z*hy*hy*pow(hz,4)
              + -2520./34560. * pow(x,8)*hy*hy*pow(hz,4)
              + -70560./34560. * x*x*pow(z,6)*hy*hy*pow(hz,4)
              + -23625./34560. * pow(y,4)*pow(z,4)*hy*hy*pow(hz,4)
              + 49455./34560. * pow(y,6)*z*z*hy*hy*pow(hz,4)
              + -2520./34560. * pow(y,8)*hy*hy*pow(hz,4)
              + -70560./34560. * y*y*pow(z,6)*hy*hy*pow(hz,4)
              + 5040./34560. * pow(z,8)*hy*hy*pow(hz,4)
              + 1200150./34560. * x*x*pow(y,4)*z*z*pow(hy,4)*hz*hz
              + -23625./34560. * pow(x,4)*pow(y,4)*pow(hy,4)*hz*hz
              + -70560./34560. * x*x*pow(y,6)*pow(hy,4)*hz*hz
              + -600075./34560. * pow(x,4)*y*y*z*z*pow(hy,4)*hz*hz
              + 49455./34560. * pow(x,6)*y*y*pow(hy,4)*hz*hz
              + -600075./34560. * x*x*y*y*pow(z,4)*pow(hy,4)*hz*hz
              + 21105./34560. * pow(x,6)*z*z*pow(hy,4)*hz*hz
              + -2520./34560. * pow(x,8)*pow(hy,4)*hz*hz
              + 47250./34560. * pow(x,4)*pow(z,4)*pow(hy,4)*hz*hz
              + 21105./34560. * x*x*pow(z,6)*pow(hy,4)*hz*hz
              + -70560./34560. * pow(y,6)*z*z*pow(hy,4)*hz*hz
              + 5040./34560. * pow(y,8)*pow(hy,4)*hz*hz
              + -23625./34560. * pow(y,4)*pow(z,4)*pow(hy,4)*hz*hz
              + 49455./34560. * y*y*pow(z,6)*pow(hy,4)*hz*hz
              + -2520./34560. * pow(z,8)*pow(hy,4)*hz*hz
              + 509040./161280. * x*x*pow(y,6)*pow(hy,6)
              + -1096200./161280. * pow(x,4)*pow(y,4)*pow(hy,6)
              + -1058400./161280. * x*x*pow(y,4)*z*z*pow(hy,6)
              + 389025./161280. * pow(x,6)*y*y*pow(hy,6)
              + 741825./161280. * pow(x,4)*y*y*z*z*pow(hy,6)
              + 316575./161280. * x*x*y*y*pow(z,4)*pow(hy,6)
              + -12600./161280. * pow(x,8)*pow(hy,6)
              + -36225./161280. * pow(x,6)*z*z*pow(hy,6)
              + -33075./161280. * pow(x,4)*pow(z,4)*pow(hy,6)
              + -7875./161280. * x*x*pow(z,6)*pow(hy,6)
              + -20160./161280. * pow(y,8)*pow(hy,6)
              + 55440./161280. * pow(y,6)*z*z*pow(hy,6)
              + 37800./161280. * pow(y,4)*pow(z,4)*pow(hy,6)
              + -36225./161280. * y*y*pow(z,6)*pow(hy,6)
              + 1575./161280. * pow(z,8)*pow(hy,6)
              + 1119825./34560. * pow(x,4)*pow(z,4)*hx*hx*pow(hz,4)
              + -438480./34560. * pow(x,6)*z*z*hx*hx*pow(hz,4)
              + -141750./34560. * pow(x,4)*y*y*z*z*hx*hx*pow(hz,4)
              + 15120./34560. * pow(x,8)*hx*hx*pow(hz,4)
              + 15120./34560. * pow(x,6)*y*y*hx*hx*pow(hz,4)
              + -14175./34560. * pow(x,4)*pow(y,4)*hx*hx*pow(hz,4)
              + -141750./34560. * x*x*y*y*pow(z,4)*hx*hx*pow(hz,4)
              + 283500./34560. * x*x*pow(y,4)*z*z*hx*hx*pow(hz,4)
              + -13230./34560. * x*x*pow(y,6)*hx*hx*pow(hz,4)
              + -438480./34560. * x*x*pow(z,6)*hx*hx*pow(hz,4)
              + -14175./34560. * pow(y,4)*pow(z,4)*hx*hx*pow(hz,4)
              + -13230./34560. * pow(y,6)*z*z*hx*hx*pow(hz,4)
              + 945./34560. * pow(y,8)*hx*hx*pow(hz,4)
              + 15120./34560. * y*y*pow(z,6)*hx*hx*pow(hz,4)
              + 15120./34560. * pow(z,8)*hx*hx*pow(hz,4)
              + 1200150./13824. * pow(x,4)*y*y*z*z*hx*hx*hy*hy*hz*hz
              + -70560./13824. * pow(x,6)*y*y*hx*hx*hy*hy*hz*hz
              + -23625./13824. * pow(x,4)*pow(y,4)*hx*hx*hy*hy*hz*hz
              + -70560./13824. * pow(x,6)*z*z*hx*hx*hy*hy*hz*hz
              + 5040./13824. * pow(x,8)*hx*hx*hy*hy*hz*hz
              + -23625./13824. * pow(x,4)*pow(z,4)*hx*hx*hy*hy*hz*hz
              + -600075./13824. * x*x*pow(y,4)*z*z*hx*hx*hy*hy*hz*hz
              + 49455./13824. * x*x*pow(y,6)*hx*hx*hy*hy*hz*hz
              + -600075./13824. * x*x*y*y*pow(z,4)*hx*hx*hy*hy*hz*hz
              + 49455./13824. * x*x*pow(z,6)*hx*hx*hy*hy*hz*hz
              + 21105./13824. * pow(y,6)*z*z*hx*hx*hy*hy*hz*hz
              + -2520./13824. * pow(y,8)*hx*hx*hy*hy*hz*hz
              + 47250./13824. * pow(y,4)*pow(z,4)*hx*hx*hy*hy*hz*hz
              + 21105./13824. * y*y*pow(z,6)*hx*hx*hy*hy*hz*hz
              + -2520./13824. * pow(z,8)*hx*hx*hy*hy*hz*hz
              + 1119825./34560. * pow(x,4)*pow(y,4)*hx*hx*pow(hy,4)
              + -438480./34560. * pow(x,6)*y*y*hx*hx*pow(hy,4)
              + -141750./34560. * pow(x,4)*y*y*z*z*hx*hx*pow(hy,4)
              + 15120./34560. * pow(x,8)*hx*hx*pow(hy,4)
              + 15120./34560. * pow(x,6)*z*z*hx*hx*pow(hy,4)
              + -14175./34560. * pow(x,4)*pow(z,4)*hx*hx*pow(hy,4)
              + -438480./34560. * x*x*pow(y,6)*hx*hx*pow(hy,4)
              + -141750./34560. * x*x*pow(y,4)*z*z*hx*hx*pow(hy,4)
              + 283500./34560. * x*x*y*y*pow(z,4)*hx*hx*pow(hy,4)
              + -13230./34560. * x*x*pow(z,6)*hx*hx*pow(hy,4)
              + 15120./34560. * pow(y,8)*hx*hx*pow(hy,4)
              + 15120./34560. * pow(y,6)*z*z*hx*hx*pow(hy,4)
              + -14175./34560. * pow(y,4)*pow(z,4)*hx*hx*pow(hy,4)
              + -13230./34560. * y*y*pow(z,6)*hx*hx*pow(hy,4)
              + 945./34560. * pow(z,8)*hx*hx*pow(hy,4)
              + 509040./34560. * pow(x,6)*z*z*pow(hx,4)*hz*hz
              + -20160./34560. * pow(x,8)*pow(hx,4)*hz*hz
              + 55440./34560. * pow(x,6)*y*y*pow(hx,4)*hz*hz
              + -1058400./34560. * pow(x,4)*y*y*z*z*pow(hx,4)*hz*hz
              + 37800./34560. * pow(x,4)*pow(y,4)*pow(hx,4)*hz*hz
              + -1096200./34560. * pow(x,4)*pow(z,4)*pow(hx,4)*hz*hz
              + 316575./34560. * x*x*pow(y,4)*z*z*pow(hx,4)*hz*hz
              + -36225./34560. * x*x*pow(y,6)*pow(hx,4)*hz*hz
              + 741825./34560. * x*x*y*y*pow(z,4)*pow(hx,4)*hz*hz
              + 389025./34560. * x*x*pow(z,6)*pow(hx,4)*hz*hz
              + -7875./34560. * pow(y,6)*z*z*pow(hx,4)*hz*hz
              + 1575./34560. * pow(y,8)*pow(hx,4)*hz*hz
              + -33075./34560. * pow(y,4)*pow(z,4)*pow(hx,4)*hz*hz
              + -36225./34560. * y*y*pow(z,6)*pow(hx,4)*hz*hz
              + -12600./34560. * pow(z,8)*pow(hx,4)*hz*hz
              + 509040./34560. * pow(x,6)*y*y*pow(hx,4)*hy*hy
              + -20160./34560. * pow(x,8)*pow(hx,4)*hy*hy
              + 55440./34560. * pow(x,6)*z*z*pow(hx,4)*hy*hy
              + -1096200./34560. * pow(x,4)*pow(y,4)*pow(hx,4)*hy*hy
              + -1058400./34560. * pow(x,4)*y*y*z*z*pow(hx,4)*hy*hy
              + 37800./34560. * pow(x,4)*pow(z,4)*pow(hx,4)*hy*hy
              + 389025./34560. * x*x*pow(y,6)*pow(hx,4)*hy*hy
              + 741825./34560. * x*x*pow(y,4)*z*z*pow(hx,4)*hy*hy
              + 316575./34560. * x*x*y*y*pow(z,4)*pow(hx,4)*hy*hy
              + -36225./34560. * x*x*pow(z,6)*pow(hx,4)*hy*hy
              + -12600./34560. * pow(y,8)*pow(hx,4)*hy*hy
              + -36225./34560. * pow(y,6)*z*z*pow(hx,4)*hy*hy
              + -33075./34560. * pow(y,4)*pow(z,4)*pow(hx,4)*hy*hy
              + -7875./34560. * y*y*pow(z,6)*pow(hx,4)*hy*hy
              + 1575./34560. * pow(z,8)*pow(hx,4)*hy*hy
              + 40320./161280. * pow(x,8)*pow(hx,6)
              + -564480./161280. * pow(x,6)*y*y*pow(hx,6)
              + -564480./161280. * pow(x,6)*z*z*pow(hx,6)
              + 1058400./161280. * pow(x,4)*pow(y,4)*pow(hx,6)
              + 2116800./161280. * pow(x,4)*y*y*z*z*pow(hx,6)
              + 1058400./161280. * pow(x,4)*pow(z,4)*pow(hx,6)
              + -352800./161280. * x*x*pow(y,6)*pow(hx,6)
              + -1058400./161280. * x*x*pow(y,4)*z*z*pow(hx,6)
              + -1058400./161280. * x*x*y*y*pow(z,4)*pow(hx,6)
              + -352800./161280. * x*x*pow(z,6)*pow(hx,6)
              + 11025./161280. * pow(y,8)*pow(hx,6)
              + 44100./161280. * pow(y,6)*z*z*pow(hx,6)
              + 66150./161280. * pow(y,4)*pow(z,4)*pow(hx,6)
              + 44100./161280. * y*y*pow(z,6)*pow(hx,6)
              + 11025./161280. * pow(z,8)*pow(hx,6))
              / pow(R,17);

    case 7:
      result += (5220./2880. * x*x*pow(z,4)*pow(hz,4)
              + -4545./2880. * pow(x,4)*z*z*pow(hz,4)
              + -4050./2880. * x*x*y*y*z*z*pow(hz,4)
              + 270./2880. * pow(x,6)*pow(hz,4)
              + 495./2880. * pow(x,4)*y*y*pow(hz,4)
              + 180./2880. * x*x*pow(y,4)*pow(hz,4)
              + 180./2880. * y*y*pow(z,4)*pow(hz,4)
              + 495./2880. * pow(y,4)*z*z*pow(hz,4)
              + -45./2880. * pow(y,6)*pow(hz,4)
              + -360./2880. * pow(z,6)*pow(hz,4)
              + 8100./1152. * x*x*y*y*z*z*hy*hy*hz*hz
              + -675./1152. * pow(x,4)*y*y*hy*hy*hz*hz
              + -675./1152. * x*x*pow(y,4)*hy*hy*hz*hz
              + -675./1152. * pow(x,4)*z*z*hy*hy*hz*hz
              + 90./1152. * pow(x,6)*hy*hy*hz*hz
              + -675./1152. * x*x*pow(z,4)*hy*hy*hz*hz
              + -675./1152. * pow(y,4)*z*z*hy*hy*hz*hz
              + 90./1152. * pow(y,6)*hy*hy*hz*hz
              + -675./1152. * y*y*pow(z,4)*hy*hy*hz*hz
              + 90./1152. * pow(z,6)*hy*hy*hz*hz
              + 5220./2880. * x*x*pow(y,4)*pow(hy,4)
              + -4545./2880. * pow(x,4)*y*y*pow(hy,4)
              + -4050./2880. * x*x*y*y*z*z*pow(hy,4)
              + 270./2880. * pow(x,6)*pow(hy,4)
              + 495./2880. * pow(x,4)*z*z*pow(hy,4)
              + 180./2880. * x*x*pow(z,4)*pow(hy,4)
              + -360./2880. * pow(y,6)*pow(hy,4)
              + 180./2880. * pow(y,4)*z*z*pow(hy,4)
              + 495./2880. * y*y*pow(z,4)*pow(hy,4)
              + -45./2880. * pow(z,6)*pow(hy,4)
              + 5220./1152. * pow(x,4)*z*z*hx*hx*hz*hz
              + -360./1152. * pow(x,6)*hx*hx*hz*hz
              + 180./1152. * pow(x,4)*y*y*hx*hx*hz*hz
              + -4050./1152. * x*x*y*y*z*z*hx*hx*hz*hz
              + 495./1152. * x*x*pow(y,4)*hx*hx*hz*hz
              + -4545./1152. * x*x*pow(z,4)*hx*hx*hz*hz
              + 180./1152. * pow(y,4)*z*z*hx*hx*hz*hz
              + -45./1152. * pow(y,6)*hx*hx*hz*hz
              + 495./1152. * y*y*pow(z,4)*hx*hx*hz*hz
              + 270./1152. * pow(z,6)*hx*hx*hz*hz
              + 5220./1152. * pow(x,4)*y*y*hx*hx*hy*hy
              + -360./1152. * pow(x,6)*hx*hx*hy*hy
              + 180./1152. * pow(x,4)*z*z*hx*hx*hy*hy
              + -4545./1152. * x*x*pow(y,4)*hx*hx*hy*hy
              + -4050./1152. * x*x*y*y*z*z*hx*hx*hy*hy
              + 495./1152. * x*x*pow(z,4)*hx*hx*hy*hy
              + 270./1152. * pow(y,6)*hx*hx*hy*hy
              + 495./1152. * pow(y,4)*z*z*hx*hx*hy*hy
              + 180./1152. * y*y*pow(z,4)*hx*hx*hy*hy
              + -45./1152. * pow(z,6)*hx*hx*hy*hy
              + 720./2880. * pow(x,6)*pow(hx,4)
              + -5400./2880. * pow(x,4)*y*y*pow(hx,4)
              + -5400./2880. * pow(x,4)*z*z*pow(hx,4)
              + 4050./2880. * x*x*pow(y,4)*pow(hx,4)
              + 8100./2880. * x*x*y*y*z*z*pow(hx,4)
              + 4050./2880. * x*x*pow(z,4)*pow(hx,4)
              + -225./2880. * pow(y,6)*pow(hx,4)
              + -675./2880. * pow(y,4)*z*z*pow(hx,4)
              + -675./2880. * y*y*pow(z,4)*pow(hx,4)
              + -225./2880. * pow(z,6)*pow(hx,4))
              / pow(R,13);

    case 5:
      result += (81./96. * x*x*z*z*hz*hz
              + -12./96. * pow(x,4)*hz*hz
              + -9./96. * x*x*y*y*hz*hz
              + -9./96. * y*y*z*z*hz*hz
              + 3./96. * pow(y,4)*hz*hz
              + -12./96. * pow(z,4)*hz*hz
              + 81./96. * x*x*y*y*hy*hy
              + -12./96. * pow(x,4)*hy*hy
              + -9./96. * x*x*z*z*hy*hy
              + -12./96. * pow(y,4)*hy*hy
              + -9./96. * y*y*z*z*hy*hy
              + 3./96. * pow(z,4)*hy*hy
              + 24./96. * pow(x,4)*hx*hx
              + -72./96. * x*x*y*y*hx*hx
              + -72./96. * x*x*z*z*hx*hx
              + 9./96. * pow(y,4)*hx*hx
              + 18./96. * y*y*z*z*hx*hx
              + 9./96. * pow(z,4)*hx*hx)
              / pow(R,9);

    case 3:
      result += (2./8. * x*x
              + -1./8. * y*y
              + -1./8. * z*z)
              / pow(R,5);
  }
  
  // multiply with prefactor
  result *= -2 * cellsize.x * cellsize.y * cellsize.z / M_PI;

  return result;
}

__host__ __device__ real calcAsymptoticNxy(int3 idx, real3 cellsize, int order) {
  double hx = cellsize.x;
  double hy = cellsize.y;
  double hz = cellsize.z;
  double x = idx.x * hx;
  double y = idx.y * hy;
  double z = idx.z * hz;
  double R = sqrt(x*x + y*y + z*z);
  
  double result = 0;

  if (!(order % 2)) {order -= 1;}
  
  switch(order) {
    case 11:
      result += (59875200./14515200. * x*y*pow(z,8)*pow(hz,8)
              + -279417600./14515200. * pow(x,3)*y*pow(z,6)*pow(hz,8)
              + -279417600./14515200. * x*pow(y,3)*pow(z,6)*pow(hz,8)
              + 261954000./14515200. * pow(x,5)*y*pow(z,4)*pow(hz,8)
              + 523908000./14515200. * pow(x,3)*pow(y,3)*pow(z,4)*pow(hz,8)
              + 261954000./14515200. * x*pow(y,5)*pow(z,4)*pow(hz,8)
              + -52390800./14515200. * pow(x,7)*y*z*z*pow(hz,8)
              + -157172400./14515200. * pow(x,5)*pow(y,3)*z*z*pow(hz,8)
              + -157172400./14515200. * pow(x,3)*pow(y,5)*z*z*pow(hz,8)
              + -52390800./14515200. * x*pow(y,7)*z*z*pow(hz,8)
              + 1091475./14515200. * pow(x,9)*y*pow(hz,8)
              + 4365900./14515200. * pow(x,7)*pow(y,3)*pow(hz,8)
              + 6548850./14515200. * pow(x,5)*pow(y,5)*pow(hz,8)
              + 4365900./14515200. * pow(x,3)*pow(y,7)*pow(hz,8)
              + 1091475./14515200. * x*pow(y,9)*pow(hz,8)
              + 253222200./1935360. * x*pow(y,3)*pow(z,6)*hy*hy*pow(hz,6)
              + -261954000./1935360. * pow(x,3)*pow(y,3)*pow(z,4)*hy*hy*pow(hz,6)
              + -301247100./1935360. * x*pow(y,5)*pow(z,4)*hy*hy*pow(hz,6)
              + 36018675./1935360. * pow(x,5)*pow(y,3)*z*z*hy*hy*pow(hz,6)
              + 121153725./1935360. * pow(x,3)*pow(y,5)*z*z*hy*hy*pow(hz,6)
              + 68762925./1935360. * x*pow(y,7)*z*z*hy*hy*pow(hz,6)
              + -155925./1935360. * pow(x,7)*pow(y,3)*hy*hy*pow(hz,6)
              + -3274425./1935360. * pow(x,5)*pow(y,5)*hy*hy*pow(hz,6)
              + -4209975./1935360. * pow(x,3)*pow(y,7)*hy*hy*pow(hz,6)
              + -1559250./1935360. * x*pow(y,9)*hy*hy*pow(hz,6)
              + 26195400./1935360. * pow(x,3)*y*pow(z,6)*hy*hy*pow(hz,6)
              + 39293100./1935360. * pow(x,5)*y*pow(z,4)*hy*hy*pow(hz,6)
              + -16372125./1935360. * pow(x,7)*y*z*z*hy*hy*pow(hz,6)
              + 467775./1935360. * pow(x,9)*y*hy*hy*pow(hz,6)
              + -29937600./1935360. * x*y*pow(z,8)*hy*hy*pow(hz,6)
              + 337265775./1036800. * x*pow(y,5)*pow(z,4)*pow(hy,4)*pow(hz,4)
              + -6548850./1036800. * pow(x,3)*pow(y,5)*z*z*pow(hy,4)*pow(hz,4)
              + -95426100./1036800. * x*pow(y,7)*z*z*pow(hy,4)*pow(hz,4)
              + -3274425./1036800. * pow(x,5)*pow(y,5)*pow(hy,4)*pow(hz,4)
              + 1871100./1036800. * pow(x,3)*pow(y,7)*pow(hy,4)*pow(hz,4)
              + 2494800./1036800. * x*pow(y,9)*pow(hy,4)*pow(hz,4)
              + -120062250./1036800. * pow(x,3)*pow(y,3)*pow(z,4)*pow(hy,4)*pow(hz,4)
              + 78586200./1036800. * pow(x,5)*pow(y,3)*z*z*pow(hy,4)*pow(hz,4)
              + -2182950./1036800. * pow(x,7)*pow(y,3)*pow(hy,4)*pow(hz,4)
              + -200831400./1036800. * x*pow(y,3)*pow(z,6)*pow(hy,4)*pow(hz,4)
              + -3274425./1036800. * pow(x,5)*y*pow(z,4)*pow(hy,4)*pow(hz,4)
              + -10291050./1036800. * pow(x,7)*y*z*z*pow(hy,4)*pow(hz,4)
              + 467775./1036800. * pow(x,9)*y*pow(hy,4)*pow(hz,4)
              + 26195400./1036800. * pow(x,3)*y*pow(z,6)*pow(hy,4)*pow(hz,4)
              + 18711000./1036800. * x*y*pow(z,8)*pow(hy,4)*pow(hz,4)
              + 145152000./1935360. * x*pow(y,7)*z*z*pow(hy,6)*hz*hz
              + 11982600./1935360. * pow(x,3)*pow(y,7)*pow(hy,6)*hz*hz
              + -4989600./1935360. * x*pow(y,9)*pow(hy,6)*hz*hz
              + -301644000./1935360. * pow(x,3)*pow(y,5)*z*z*pow(hy,6)*hz*hz
              + 5641650./1935360. * pow(x,5)*pow(y,5)*pow(hy,6)*hz*hz
              + -366820650./1935360. * x*pow(y,5)*pow(z,4)*pow(hy,6)*hz*hz
              + 160588575./1935360. * pow(x,5)*pow(y,3)*z*z*pow(hy,6)*hz*hz
              + -10163475./1935360. * pow(x,7)*pow(y,3)*pow(hy,6)*hz*hz
              + 218139075./1935360. * pow(x,3)*pow(y,3)*pow(z,4)*pow(hy,6)*hz*hz
              + 211250025./1935360. * x*pow(y,3)*pow(z,6)*pow(hy,6)*hz*hz
              + -12885075./1935360. * pow(x,7)*y*z*z*pow(hy,6)*hz*hz
              + 1167075./1935360. * pow(x,9)*y*pow(hy,6)*hz*hz
              + -28080675./1935360. * pow(x,5)*y*pow(z,4)*pow(hy,6)*hz*hz
              + -32115825./1935360. * pow(x,3)*y*pow(z,6)*pow(hy,6)*hz*hz
              + -18087300./1935360. * x*y*pow(z,8)*pow(hy,6)*hz*hz
              + 19958400./14515200. * x*pow(y,9)*pow(hy,8)
              + -179625600./14515200. * pow(x,3)*pow(y,7)*pow(hy,8)
              + -179625600./14515200. * x*pow(y,7)*z*z*pow(hy,8)
              + 314344800./14515200. * pow(x,5)*pow(y,5)*pow(hy,8)
              + 577092600./14515200. * pow(x,3)*pow(y,5)*z*z*pow(hy,8)
              + 365941800./14515200. * x*pow(y,5)*pow(z,4)*pow(hy,8)
              + -130977000./14515200. * pow(x,7)*pow(y,3)*pow(hy,8)
              + -356359500./14515200. * pow(x,5)*pow(y,3)*z*z*pow(hy,8)
              + -392931000./14515200. * pow(x,3)*pow(y,3)*pow(z,4)*pow(hy,8)
              + -167548500./14515200. * x*pow(y,3)*pow(z,6)*pow(hy,8)
              + 9823275./14515200. * pow(x,9)*y*pow(hy,8)
              + 35891100./14515200. * pow(x,7)*y*z*z*pow(hy,8)
              + 55537650./14515200. * pow(x,5)*y*pow(z,4)*pow(hy,8)
              + 42695100./14515200. * pow(x,3)*y*pow(z,6)*pow(hy,8)
              + 13225275./14515200. * x*y*pow(z,8)*pow(hy,8)
              + 253222200./1935360. * pow(x,3)*y*pow(z,6)*hx*hx*pow(hz,6)
              + -301247100./1935360. * pow(x,5)*y*pow(z,4)*hx*hx*pow(hz,6)
              + -261954000./1935360. * pow(x,3)*pow(y,3)*pow(z,4)*hx*hx*pow(hz,6)
              + 68762925./1935360. * pow(x,7)*y*z*z*hx*hx*pow(hz,6)
              + 121153725./1935360. * pow(x,5)*pow(y,3)*z*z*hx*hx*pow(hz,6)
              + 36018675./1935360. * pow(x,3)*pow(y,5)*z*z*hx*hx*pow(hz,6)
              + -1559250./1935360. * pow(x,9)*y*hx*hx*pow(hz,6)
              + -4209975./1935360. * pow(x,7)*pow(y,3)*hx*hx*pow(hz,6)
              + -3274425./1935360. * pow(x,5)*pow(y,5)*hx*hx*pow(hz,6)
              + -155925./1935360. * pow(x,3)*pow(y,7)*hx*hx*pow(hz,6)
              + 26195400./1935360. * x*pow(y,3)*pow(z,6)*hx*hx*pow(hz,6)
              + 39293100./1935360. * x*pow(y,5)*pow(z,4)*hx*hx*pow(hz,6)
              + -16372125./1935360. * x*pow(y,7)*z*z*hx*hx*pow(hz,6)
              + 467775./1935360. * x*pow(y,9)*hx*hx*pow(hz,6)
              + -29937600./1935360. * x*y*pow(z,8)*hx*hx*pow(hz,6)
              + 382016250./414720. * pow(x,3)*pow(y,3)*pow(z,4)*hx*hx*hy*hy*pow(hz,4)
              + -114604875./414720. * pow(x,5)*pow(y,3)*z*z*hx*hx*hy*hy*pow(hz,4)
              + -114604875./414720. * pow(x,3)*pow(y,5)*z*z*hx*hx*hy*hy*pow(hz,4)
              + 2338875./414720. * pow(x,7)*pow(y,3)*hx*hx*hy*hy*pow(hz,4)
              + 6548850./414720. * pow(x,5)*pow(y,5)*hx*hx*hy*hy*pow(hz,4)
              + 2338875./414720. * pow(x,3)*pow(y,7)*hx*hx*hy*hy*pow(hz,4)
              + -36018675./414720. * pow(x,5)*y*pow(z,4)*hx*hx*hy*hy*pow(hz,4)
              + 26663175./414720. * pow(x,7)*y*z*z*hx*hx*hy*hy*pow(hz,4)
              + -935550./414720. * pow(x,9)*y*hx*hx*hy*hy*pow(hz,4)
              + -52390800./414720. * pow(x,3)*y*pow(z,6)*hx*hx*hy*hy*pow(hz,4)
              + -36018675./414720. * x*pow(y,5)*pow(z,4)*hx*hx*hy*hy*pow(hz,4)
              + 26663175./414720. * x*pow(y,7)*z*z*hx*hx*hy*hy*pow(hz,4)
              + -935550./414720. * x*pow(y,9)*hx*hx*hy*hy*pow(hz,4)
              + -52390800./414720. * x*pow(y,3)*pow(z,6)*hx*hx*hy*hy*pow(hz,4)
              + 11226600./414720. * x*y*pow(z,8)*hx*hx*hy*hy*pow(hz,4)
              + 333991350./414720. * pow(x,3)*pow(y,5)*z*z*hx*hx*pow(hy,4)*hz*hz
              + -3274425./414720. * pow(x,5)*pow(y,5)*hx*hx*pow(hy,4)*hz*hz
              + -14345100./414720. * pow(x,3)*pow(y,7)*hx*hx*pow(hy,4)*hz*hz
              + -219386475./414720. * pow(x,5)*pow(y,3)*z*z*hx*hx*pow(hy,4)*hz*hz
              + 12006225./414720. * pow(x,7)*pow(y,3)*hx*hx*pow(hy,4)*hz*hz
              + -191008125./414720. * pow(x,3)*pow(y,3)*pow(z,4)*hx*hx*pow(hy,4)*hz*hz
              + 20114325./414720. * pow(x,7)*y*z*z*hx*hx*pow(hy,4)*hz*hz
              + -1559250./414720. * pow(x,9)*y*hx*hx*pow(hy,4)*hz*hz
              + 39293100./414720. * pow(x,5)*y*pow(z,4)*hx*hx*pow(hy,4)*hz*hz
              + 12006225./414720. * pow(x,3)*y*pow(z,6)*hx*hx*pow(hy,4)*hz*hz
              + -46777500./414720. * x*pow(y,7)*z*z*hx*hx*pow(hy,4)*hz*hz
              + 2494800./414720. * x*pow(y,9)*hx*hx*pow(hy,4)*hz*hz
              + -3274425./414720. * x*pow(y,5)*pow(z,4)*hx*hx*pow(hy,4)*hz*hz
              + 40384575./414720. * x*pow(y,3)*pow(z,6)*hx*hx*pow(hy,4)*hz*hz
              + -5613300./414720. * x*y*pow(z,8)*hx*hx*pow(hy,4)*hz*hz
              + 167151600./1935360. * pow(x,3)*pow(y,7)*hx*hx*pow(hy,6)
              + -320893650./1935360. * pow(x,5)*pow(y,5)*hx*hx*pow(hy,6)
              + -301247100./1935360. * pow(x,3)*pow(y,5)*z*z*hx*hx*pow(hy,6)
              + 140800275./1935360. * pow(x,7)*pow(y,3)*hx*hx*pow(hy,6)
              + 252130725./1935360. * pow(x,5)*pow(y,3)*z*z*hx*hx*pow(hy,6)
              + 81860625./1935360. * pow(x,3)*pow(y,3)*pow(z,4)*hx*hx*pow(hy,6)
              + -10914750./1935360. * pow(x,9)*y*hx*hx*pow(hy,6)
              + -29469825./1935360. * pow(x,7)*y*z*z*hx*hx*pow(hy,6)
              + -22920975./1935360. * pow(x,5)*y*pow(z,4)*hx*hx*pow(hy,6)
              + -1091475./1935360. * pow(x,3)*y*pow(z,6)*hx*hx*pow(hy,6)
              + -14968800./1935360. * x*pow(y,9)*hx*hx*pow(hy,6)
              + 37422000./1935360. * x*pow(y,7)*z*z*hx*hx*pow(hy,6)
              + 19646550./1935360. * x*pow(y,5)*pow(z,4)*hx*hx*pow(hy,6)
              + -29469825./1935360. * x*pow(y,3)*pow(z,6)*hx*hx*pow(hy,6)
              + 3274425./1935360. * x*y*pow(z,8)*hx*hx*pow(hy,6)
              + 337265775./1036800. * pow(x,5)*y*pow(z,4)*pow(hx,4)*pow(hz,4)
              + -95426100./1036800. * pow(x,7)*y*z*z*pow(hx,4)*pow(hz,4)
              + -6548850./1036800. * pow(x,5)*pow(y,3)*z*z*pow(hx,4)*pow(hz,4)
              + 2494800./1036800. * pow(x,9)*y*pow(hx,4)*pow(hz,4)
              + 1871100./1036800. * pow(x,7)*pow(y,3)*pow(hx,4)*pow(hz,4)
              + -3274425./1036800. * pow(x,5)*pow(y,5)*pow(hx,4)*pow(hz,4)
              + -120062250./1036800. * pow(x,3)*pow(y,3)*pow(z,4)*pow(hx,4)*pow(hz,4)
              + 78586200./1036800. * pow(x,3)*pow(y,5)*z*z*pow(hx,4)*pow(hz,4)
              + -2182950./1036800. * pow(x,3)*pow(y,7)*pow(hx,4)*pow(hz,4)
              + -200831400./1036800. * pow(x,3)*y*pow(z,6)*pow(hx,4)*pow(hz,4)
              + -3274425./1036800. * x*pow(y,5)*pow(z,4)*pow(hx,4)*pow(hz,4)
              + -10291050./1036800. * x*pow(y,7)*z*z*pow(hx,4)*pow(hz,4)
              + 467775./1036800. * x*pow(y,9)*pow(hx,4)*pow(hz,4)
              + 26195400./1036800. * x*pow(y,3)*pow(z,6)*pow(hx,4)*pow(hz,4)
              + 18711000./1036800. * x*y*pow(z,8)*pow(hx,4)*pow(hz,4)
              + 333991350./414720. * pow(x,5)*pow(y,3)*z*z*pow(hx,4)*hy*hy*hz*hz
              + -14345100./414720. * pow(x,7)*pow(y,3)*pow(hx,4)*hy*hy*hz*hz
              + -3274425./414720. * pow(x,5)*pow(y,5)*pow(hx,4)*hy*hy*hz*hz
              + -46777500./414720. * pow(x,7)*y*z*z*pow(hx,4)*hy*hy*hz*hz
              + 2494800./414720. * pow(x,9)*y*pow(hx,4)*hy*hy*hz*hz
              + -3274425./414720. * pow(x,5)*y*pow(z,4)*pow(hx,4)*hy*hy*hz*hz
              + -219386475./414720. * pow(x,3)*pow(y,5)*z*z*pow(hx,4)*hy*hy*hz*hz
              + 12006225./414720. * pow(x,3)*pow(y,7)*pow(hx,4)*hy*hy*hz*hz
              + -191008125./414720. * pow(x,3)*pow(y,3)*pow(z,4)*pow(hx,4)*hy*hy*hz*hz
              + 40384575./414720. * pow(x,3)*y*pow(z,6)*pow(hx,4)*hy*hy*hz*hz
              + 20114325./414720. * x*pow(y,7)*z*z*pow(hx,4)*hy*hy*hz*hz
              + -1559250./414720. * x*pow(y,9)*pow(hx,4)*hy*hy*hz*hz
              + 39293100./414720. * x*pow(y,5)*pow(z,4)*pow(hx,4)*hy*hy*hz*hz
              + 12006225./414720. * x*pow(y,3)*pow(z,6)*pow(hx,4)*hy*hy*hz*hz
              + -5613300./414720. * x*y*pow(z,8)*pow(hx,4)*hy*hy*hz*hz
              + 324168075./1036800. * pow(x,5)*pow(y,5)*pow(hx,4)*pow(hy,4)
              + -152806500./1036800. * pow(x,7)*pow(y,3)*pow(hx,4)*pow(hy,4)
              + -32744250./1036800. * pow(x,5)*pow(y,3)*z*z*pow(hx,4)*pow(hy,4)
              + 12474000./1036800. * pow(x,9)*y*pow(hx,4)*pow(hy,4)
              + 9355500./1036800. * pow(x,7)*y*z*z*pow(hx,4)*pow(hy,4)
              + -16372125./1036800. * pow(x,5)*y*pow(z,4)*pow(hx,4)*pow(hy,4)
              + -152806500./1036800. * pow(x,3)*pow(y,7)*pow(hx,4)*pow(hy,4)
              + -32744250./1036800. * pow(x,3)*pow(y,5)*z*z*pow(hx,4)*pow(hy,4)
              + 109147500./1036800. * pow(x,3)*pow(y,3)*pow(z,4)*pow(hx,4)*pow(hy,4)
              + -10914750./1036800. * pow(x,3)*y*pow(z,6)*pow(hx,4)*pow(hy,4)
              + 12474000./1036800. * x*pow(y,9)*pow(hx,4)*pow(hy,4)
              + 9355500./1036800. * x*pow(y,7)*z*z*pow(hx,4)*pow(hy,4)
              + -16372125./1036800. * x*pow(y,5)*pow(z,4)*pow(hx,4)*pow(hy,4)
              + -10914750./1036800. * x*pow(y,3)*pow(z,6)*pow(hx,4)*pow(hy,4)
              + 2338875./1036800. * x*y*pow(z,8)*pow(hx,4)*pow(hy,4)
              + 145152000./1935360. * pow(x,7)*y*z*z*pow(hx,6)*hz*hz
              + -4989600./1935360. * pow(x,9)*y*pow(hx,6)*hz*hz
              + 11982600./1935360. * pow(x,7)*pow(y,3)*pow(hx,6)*hz*hz
              + -301644000./1935360. * pow(x,5)*pow(y,3)*z*z*pow(hx,6)*hz*hz
              + 5641650./1935360. * pow(x,5)*pow(y,5)*pow(hx,6)*hz*hz
              + -366820650./1935360. * pow(x,5)*y*pow(z,4)*pow(hx,6)*hz*hz
              + 160588575./1935360. * pow(x,3)*pow(y,5)*z*z*pow(hx,6)*hz*hz
              + -10163475./1935360. * pow(x,3)*pow(y,7)*pow(hx,6)*hz*hz
              + 218139075./1935360. * pow(x,3)*pow(y,3)*pow(z,4)*pow(hx,6)*hz*hz
              + 211250025./1935360. * pow(x,3)*y*pow(z,6)*pow(hx,6)*hz*hz
              + -12885075./1935360. * x*pow(y,7)*z*z*pow(hx,6)*hz*hz
              + 1167075./1935360. * x*pow(y,9)*pow(hx,6)*hz*hz
              + -28080675./1935360. * x*pow(y,5)*pow(z,4)*pow(hx,6)*hz*hz
              + -32115825./1935360. * x*pow(y,3)*pow(z,6)*pow(hx,6)*hz*hz
              + -18087300./1935360. * x*y*pow(z,8)*pow(hx,6)*hz*hz
              + 167151600./1935360. * pow(x,7)*pow(y,3)*pow(hx,6)*hy*hy
              + -14968800./1935360. * pow(x,9)*y*pow(hx,6)*hy*hy
              + 35947800./1935360. * pow(x,7)*y*z*z*pow(hx,6)*hy*hy
              + -320893650./1935360. * pow(x,5)*pow(y,5)*pow(hx,6)*hy*hy
              + -274201200./1935360. * pow(x,5)*pow(y,3)*z*z*pow(hx,6)*hy*hy
              + 4167450./1935360. * pow(x,5)*y*pow(z,4)*pow(hx,6)*hy*hy
              + 140800275./1935360. * pow(x,3)*pow(y,7)*pow(hx,6)*hy*hy
              + 194977125./1935360. * pow(x,3)*pow(y,5)*z*z*pow(hx,6)*hy*hy
              + 176492925./1935360. * pow(x,3)*pow(y,3)*pow(z,4)*pow(hx,6)*hy*hy
              + -41546925./1935360. * pow(x,3)*y*pow(z,6)*pow(hx,6)*hy*hy
              + -10914750./1935360. * x*pow(y,9)*pow(hx,6)*hy*hy
              + -23573025./1935360. * x*pow(y,7)*z*z*pow(hx,6)*hy*hy
              + -28477575./1935360. * x*pow(y,5)*pow(z,4)*pow(hx,6)*hy*hy
              + -10617075./1935360. * x*pow(y,3)*pow(z,6)*pow(hx,6)*hy*hy
              + 5202225./1935360. * x*y*pow(z,8)*pow(hx,6)*hy*hy
              + 19958400./14515200. * pow(x,9)*y*pow(hx,8)
              + -179625600./14515200. * pow(x,7)*pow(y,3)*pow(hx,8)
              + -179625600./14515200. * pow(x,7)*y*z*z*pow(hx,8)
              + 314344800./14515200. * pow(x,5)*pow(y,5)*pow(hx,8)
              + 577092600./14515200. * pow(x,5)*pow(y,3)*z*z*pow(hx,8)
              + 365941800./14515200. * pow(x,5)*y*pow(z,4)*pow(hx,8)
              + -130977000./14515200. * pow(x,3)*pow(y,7)*pow(hx,8)
              + -356359500./14515200. * pow(x,3)*pow(y,5)*z*z*pow(hx,8)
              + -392931000./14515200. * pow(x,3)*pow(y,3)*pow(z,4)*pow(hx,8)
              + -167548500./14515200. * pow(x,3)*y*pow(z,6)*pow(hx,8)
              + 9823275./14515200. * x*pow(y,9)*pow(hx,8)
              + 35891100./14515200. * x*pow(y,7)*z*z*pow(hx,8)
              + 55537650./14515200. * x*pow(y,5)*pow(z,4)*pow(hx,8)
              + 42695100./14515200. * x*pow(y,3)*pow(z,6)*pow(hx,8)
              + 13225275./14515200. * x*y*pow(z,8)*pow(hx,8))
              / pow(R,21);

    case 9:
      result += (453600./161280. * x*y*pow(z,6)*pow(hz,6)
              + -1134000./161280. * pow(x,3)*y*pow(z,4)*pow(hz,6)
              + -1134000./161280. * x*pow(y,3)*pow(z,4)*pow(hz,6)
              + 425250./161280. * pow(x,5)*y*z*z*pow(hz,6)
              + 850500./161280. * pow(x,3)*pow(y,3)*z*z*pow(hz,6)
              + 425250./161280. * x*pow(y,5)*z*z*pow(hz,6)
              + -14175./161280. * pow(x,7)*y*pow(hz,6)
              + -42525./161280. * pow(x,5)*pow(y,3)*pow(hz,6)
              + -42525./161280. * pow(x,3)*pow(y,5)*pow(hz,6)
              + -14175./161280. * x*pow(y,7)*pow(hz,6)
              + 1190700./34560. * x*pow(y,3)*pow(z,4)*hy*hy*pow(hz,4)
              + -425250./34560. * pow(x,3)*pow(y,3)*z*z*hy*hy*pow(hz,4)
              + -586845./34560. * x*pow(y,5)*z*z*hy*hy*pow(hz,4)
              + 5670./34560. * pow(x,5)*pow(y,3)*hy*hy*pow(hz,4)
              + 36855./34560. * pow(x,3)*pow(y,5)*hy*hy*pow(hz,4)
              + 22680./34560. * x*pow(y,7)*hy*hy*pow(hz,4)
              + -56700./34560. * pow(x,3)*y*pow(z,4)*hy*hy*pow(hz,4)
              + 161595./34560. * pow(x,5)*y*z*z*hy*hy*pow(hz,4)
              + -8505./34560. * pow(x,7)*y*hy*hy*pow(hz,4)
              + -226800./34560. * x*y*pow(z,6)*hy*hy*pow(hz,4)
              + 861840./34560. * x*pow(y,5)*z*z*pow(hy,4)*hz*hz
              + 30240./34560. * pow(x,3)*pow(y,5)*pow(hy,4)*hz*hz
              + -45360./34560. * x*pow(y,7)*pow(hy,4)*hz*hz
              + -916650./34560. * pow(x,3)*pow(y,3)*z*z*pow(hy,4)*hz*hz
              + 61425./34560. * pow(x,5)*pow(y,3)*pow(hy,4)*hz*hz
              + -978075./34560. * x*pow(y,3)*pow(z,4)*pow(hy,4)*hz*hz
              + 113400./34560. * pow(x,5)*y*z*z*pow(hy,4)*hz*hz
              + -14175./34560. * pow(x,7)*y*pow(hy,4)*hz*hz
              + 269325./34560. * pow(x,3)*y*pow(z,4)*pow(hy,4)*hz*hz
              + 141750./34560. * x*y*pow(z,6)*pow(hy,4)*hz*hz
              + 181440./161280. * x*pow(y,7)*pow(hy,6)
              + -952560./161280. * pow(x,3)*pow(y,5)*pow(hy,6)
              + -952560./161280. * x*pow(y,5)*z*z*pow(hy,6)
              + 793800./161280. * pow(x,5)*pow(y,3)*pow(hy,6)
              + 1341900./161280. * pow(x,3)*pow(y,3)*z*z*pow(hy,6)
              + 1039500./161280. * x*pow(y,3)*pow(z,4)*pow(hy,6)
              + -99225./161280. * pow(x,7)*y*pow(hy,6)
              + -259875./161280. * pow(x,5)*y*z*z*pow(hy,6)
              + -297675./161280. * pow(x,3)*y*pow(z,4)*pow(hy,6)
              + -137025./161280. * x*y*pow(z,6)*pow(hy,6)
              + 1190700./34560. * pow(x,3)*y*pow(z,4)*hx*hx*pow(hz,4)
              + -586845./34560. * pow(x,5)*y*z*z*hx*hx*pow(hz,4)
              + -425250./34560. * pow(x,3)*pow(y,3)*z*z*hx*hx*pow(hz,4)
              + 22680./34560. * pow(x,7)*y*hx*hx*pow(hz,4)
              + 36855./34560. * pow(x,5)*pow(y,3)*hx*hx*pow(hz,4)
              + 5670./34560. * pow(x,3)*pow(y,5)*hx*hx*pow(hz,4)
              + -56700./34560. * x*pow(y,3)*pow(z,4)*hx*hx*pow(hz,4)
              + 161595./34560. * x*pow(y,5)*z*z*hx*hx*pow(hz,4)
              + -8505./34560. * x*pow(y,7)*hx*hx*pow(hz,4)
              + -226800./34560. * x*y*pow(z,6)*hx*hx*pow(hz,4)
              + 1341900./13824. * pow(x,3)*pow(y,3)*z*z*hx*hx*hy*hy*hz*hz
              + -67095./13824. * pow(x,5)*pow(y,3)*hx*hx*hy*hy*hz*hz
              + -67095./13824. * pow(x,3)*pow(y,5)*hx*hx*hy*hy*hz*hz
              + -274995./13824. * pow(x,5)*y*z*z*hx*hx*hy*hy*hz*hz
              + 22680./13824. * pow(x,7)*y*hx*hx*hy*hy*hz*hz
              + -212625./13824. * pow(x,3)*y*pow(z,4)*hx*hx*hy*hy*hz*hz
              + -274995./13824. * x*pow(y,5)*z*z*hx*hx*hy*hy*hz*hz
              + 22680./13824. * x*pow(y,7)*hx*hx*hy*hy*hz*hz
              + -212625./13824. * x*pow(y,3)*pow(z,4)*hx*hx*hy*hy*hz*hz
              + 85050./13824. * x*y*pow(z,6)*hx*hx*hy*hy*hz*hz
              + 922320./34560. * pow(x,3)*pow(y,5)*hx*hx*pow(hy,4)
              + -855225./34560. * pow(x,5)*pow(y,3)*hx*hx*pow(hy,4)
              + -670950./34560. * pow(x,3)*pow(y,3)*z*z*hx*hx*pow(hy,4)
              + 113400./34560. * pow(x,7)*y*hx*hx*pow(hy,4)
              + 184275./34560. * pow(x,5)*y*z*z*hx*hx*pow(hy,4)
              + 28350./34560. * pow(x,3)*y*pow(z,4)*hx*hx*pow(hy,4)
              + -136080./34560. * x*pow(y,7)*hx*hx*pow(hy,4)
              + 90720./34560. * x*pow(y,5)*z*z*hx*hx*pow(hy,4)
              + 184275./34560. * x*pow(y,3)*pow(z,4)*hx*hx*pow(hy,4)
              + -42525./34560. * x*y*pow(z,6)*hx*hx*pow(hy,4)
              + 861840./34560. * pow(x,5)*y*z*z*pow(hx,4)*hz*hz
              + -45360./34560. * pow(x,7)*y*pow(hx,4)*hz*hz
              + 30240./34560. * pow(x,5)*pow(y,3)*pow(hx,4)*hz*hz
              + -916650./34560. * pow(x,3)*pow(y,3)*z*z*pow(hx,4)*hz*hz
              + 61425./34560. * pow(x,3)*pow(y,5)*pow(hx,4)*hz*hz
              + -978075./34560. * pow(x,3)*y*pow(z,4)*pow(hx,4)*hz*hz
              + 113400./34560. * x*pow(y,5)*z*z*pow(hx,4)*hz*hz
              + -14175./34560. * x*pow(y,7)*pow(hx,4)*hz*hz
              + 269325./34560. * x*pow(y,3)*pow(z,4)*pow(hx,4)*hz*hz
              + 141750./34560. * x*y*pow(z,6)*pow(hx,4)*hz*hz
              + 922320./34560. * pow(x,5)*pow(y,3)*pow(hx,4)*hy*hy
              + -136080./34560. * pow(x,7)*y*pow(hx,4)*hy*hy
              + 90720./34560. * pow(x,5)*y*z*z*pow(hx,4)*hy*hy
              + -855225./34560. * pow(x,3)*pow(y,5)*pow(hx,4)*hy*hy
              + -670950./34560. * pow(x,3)*pow(y,3)*z*z*pow(hx,4)*hy*hy
              + 184275./34560. * pow(x,3)*y*pow(z,4)*pow(hx,4)*hy*hy
              + 113400./34560. * x*pow(y,7)*pow(hx,4)*hy*hy
              + 184275./34560. * x*pow(y,5)*z*z*pow(hx,4)*hy*hy
              + 28350./34560. * x*pow(y,3)*pow(z,4)*pow(hx,4)*hy*hy
              + -42525./34560. * x*y*pow(z,6)*pow(hx,4)*hy*hy
              + 181440./161280. * pow(x,7)*y*pow(hx,6)
              + -952560./161280. * pow(x,5)*pow(y,3)*pow(hx,6)
              + -952560./161280. * pow(x,5)*y*z*z*pow(hx,6)
              + 793800./161280. * pow(x,3)*pow(y,5)*pow(hx,6)
              + 1341900./161280. * pow(x,3)*pow(y,3)*z*z*pow(hx,6)
              + 1039500./161280. * pow(x,3)*y*pow(z,4)*pow(hx,6)
              + -99225./161280. * x*pow(y,7)*pow(hx,6)
              + -259875./161280. * x*pow(y,5)*z*z*pow(hx,6)
              + -297675./161280. * x*pow(y,3)*pow(z,4)*pow(hx,6)
              + -137025./161280. * x*y*pow(z,6)*pow(hx,6))
              / pow(R,17);

    case 7:
      result += (5040./2880. * x*y*pow(z,4)*pow(hz,4)
              + -5040./2880. * pow(x,3)*y*z*z*pow(hz,4)
              + -5040./2880. * x*pow(y,3)*z*z*pow(hz,4)
              + 315./2880. * pow(x,5)*y*pow(hz,4)
              + 630./2880. * pow(x,3)*pow(y,3)*pow(hz,4)
              + 315./2880. * x*pow(y,5)*pow(hz,4)
              + 7245./1152. * x*pow(y,3)*z*z*hy*hy*hz*hz
              + -315./1152. * pow(x,3)*pow(y,3)*hy*hy*hz*hz
              + -630./1152. * x*pow(y,5)*hy*hy*hz*hz
              + -2205./1152. * pow(x,3)*y*z*z*hy*hy*hz*hz
              + 315./1152. * pow(x,5)*y*hy*hy*hz*hz
              + -2520./1152. * x*y*pow(z,4)*hy*hy*hz*hz
              + 2520./2880. * x*pow(y,5)*pow(hy,4)
              + -6300./2880. * pow(x,3)*pow(y,3)*pow(hy,4)
              + -6300./2880. * x*pow(y,3)*z*z*pow(hy,4)
              + 1575./2880. * pow(x,5)*y*pow(hy,4)
              + 3150./2880. * pow(x,3)*y*z*z*pow(hy,4)
              + 1575./2880. * x*y*pow(z,4)*pow(hy,4)
              + 7245./1152. * pow(x,3)*y*z*z*hx*hx*hz*hz
              + -630./1152. * pow(x,5)*y*hx*hx*hz*hz
              + -315./1152. * pow(x,3)*pow(y,3)*hx*hx*hz*hz
              + -2205./1152. * x*pow(y,3)*z*z*hx*hx*hz*hz
              + 315./1152. * x*pow(y,5)*hx*hx*hz*hz
              + -2520./1152. * x*y*pow(z,4)*hx*hx*hz*hz
              + 6615./1152. * pow(x,3)*pow(y,3)*hx*hx*hy*hy
              + -1890./1152. * pow(x,5)*y*hx*hx*hy*hy
              + -945./1152. * pow(x,3)*y*z*z*hx*hx*hy*hy
              + -1890./1152. * x*pow(y,5)*hx*hx*hy*hy
              + -945./1152. * x*pow(y,3)*z*z*hx*hx*hy*hy
              + 945./1152. * x*y*pow(z,4)*hx*hx*hy*hy
              + 2520./2880. * pow(x,5)*y*pow(hx,4)
              + -6300./2880. * pow(x,3)*pow(y,3)*pow(hx,4)
              + -6300./2880. * pow(x,3)*y*z*z*pow(hx,4)
              + 1575./2880. * x*pow(y,5)*pow(hx,4)
              + 3150./2880. * x*pow(y,3)*z*z*pow(hx,4)
              + 1575./2880. * x*y*pow(z,4)*pow(hx,4))
              / pow(R,13);

    case 5:
      result += (90./96. * x*y*z*z*hz*hz
              + -15./96. * pow(x,3)*y*hz*hz
              + -15./96. * x*pow(y,3)*hz*hz
              + 60./96. * x*pow(y,3)*hy*hy
              + -45./96. * pow(x,3)*y*hy*hy
              + -45./96. * x*y*z*z*hy*hy
              + 60./96. * pow(x,3)*y*hx*hx
              + -45./96. * x*pow(y,3)*hx*hx
              + -45./96. * x*y*z*z*hx*hx)
              / pow(R,9);

    case 3:
      result += (3./8. * x*y) / pow(R,5);
  }

  // multiply with prefactor
  result *= -2 * cellsize.x * cellsize.y * cellsize.z / M_PI;

  return result;
}

// reuse Nxx and Nxy by permutating the arguments to implement the other kernel
// components
real calcAsymptoticNyy(int3 idx, real3 cs, int order) {
  return calcAsymptoticNxx({idx.y, idx.z, idx.x}, {cs.y, cs.z, cs.x}, order);
}
real calcAsymptoticNzz(int3 idx, real3 cs, int order) {
  return calcAsymptoticNxx({idx.z, idx.x, idx.y}, {cs.z, cs.x, cs.y}, order);
}
real calcAsymptoticNxz(int3 idx, real3 cs, int order) {
  return calcAsymptoticNxy({idx.x, idx.z, idx.y}, {cs.x, cs.z, cs.y}, order);
}
real calcAsymptoticNyx(int3 idx, real3 cs, int order) {
  return calcAsymptoticNxy({idx.y, idx.x, idx.z}, {cs.y, cs.x, cs.z}, order);
}
real calcAsymptoticNyz(int3 idx, real3 cs, int order) {
  return calcAsymptoticNxy({idx.y, idx.z, idx.x}, {cs.y, cs.z, cs.x}, order);
}
real calcAsymptoticNzx(int3 idx, real3 cs, int order) {
  return calcAsymptoticNxy({idx.z, idx.x, idx.y}, {cs.z, cs.x, cs.y}, order);
}
real calcAsymptoticNzy(int3 idx, real3 cs, int order) {
  return calcAsymptoticNxy({idx.z, idx.y, idx.x}, {cs.z, cs.y, cs.x}, order);
}
