C
      SUBROUTINE gs1 (f, h, beta, lam , tol,n,u)
C     
      integer N,iter,j
CF2PY INTENT(OUT) :: u
CF2PY INTENT(HIDE) :: N
CF2PY  real*8 :: f(N)
CF2PY  real*8 :: u(N)
CF2PY  real*8:: u1(N)
CF2PY  real*8:: u_old(N)
      
      real*8  f(*)
      real*8  u(*)
      real*8, allocatable :: u1(:)
      real*8,  allocatable :: u_old(:)
      real*8  nu, alpha, beta,lam, tol
      real*8  error,a,b,den, sum,err

      allocate(u1(N+2))
      allocate(u_old(N+2))
      u1(1:n+2)=0.0d0
      
      u1(2:n + 1)=f(1:n)
      iter = 0
      error = 1.0d6
      sum = 0.0d0
      err = 0.0d0
      do while (error.gt.tol) 
         iter = iter + 1
         u1(1) = u1(2)
         u1(n + 2) = u1(n+1)
         u_old(1:n+2) = u1(1:n+2)
         do j = 2,n + 1
            nu = (u1(j + 1) - u1(j - 1)) / (2.0d0 * h)
            nu = beta + nu * nu
            nu = nu * sqrt(nu)
            nu = 1.0d0 / nu
            alpha = beta * nu
            den = (2 * alpha) / (lam *h*h)
            a = 1.0 / (1.0 + 2 * den)
            b = den / (1.0 + 2 * den)
            u1(j)= a * f(j - 1) + b * (u1(j - 1) + u1(j + 1))

         end do
         if (iter.gt.10000) then
            print*,"Not convergence reach",error
            exit
          end if
         error = norm2(u1-u_old)/norm2(u1)
         !print*, iter,error
      end do
      !print*, "Numero de iteraciones: ",iter
      u(1:n) = u1(2:n + 1)

      end subroutine gs1
