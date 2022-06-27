### ====================================================
### = krig_wrapper_AJT.r
### = Alex Turner
### = 07/10/2014
### =---------------------------------------------------
### = NOTES:
### = ( 1) A wrapper for the "fields" kriging function.
### ====================================================

### Load the necessary libraries
require(fields)
require(reshape2)

### Kriging function
krig_fun <- function(x,y,z,xU,yU,lx,ly){
   # Get matrix sizes
   nX <- length(xU)
   nY <- length(yU)
   # Create maps between the input and output data
   m <- matrix(data=NA,nrow=length(x), ncol=length(y));  colnames(m)  <- (y*ly);  rownames(m)  <- (x/(120)*lx)
   mO<- matrix(data=NA,nrow=length(xU),ncol=length(yU)); colnames(mO) <- (yU*ly); rownames(mO) <- (xU/(120)*lx)
   mN<- matrix(data=NA,nrow=length(xU),ncol=length(yU)); colnames(mN) <- yU;      rownames(mN) <- xU
   m <- melt(m);  m <- m[,1:2]
   mO<- melt(mO); mO<- mO[,1:2]
   mN<- melt(mN); mN<- mN[,1:2]
   # Krig the data to the finer grid
   dat <- melt(z); names(dat) <- c("lat","alt","co2")
   options(warn=-1) # Suppress an annoying warning that we don't have a bounded minimum
   fit <- Krig(m,dat[,3],theta=1)
   res <- predict(fit,mO)
   res <- cbind(mN,res); colnames(res) <- c("lat","alt","co2")
   # Reshape to a matrix
   out <- matrix(ncol=nY,nrow=nX)
   ind <- 1
   for (i in 1:nY){
      for (j in 1:nX){
         out[j,i] <- res[ind,3]
         ind      <- ind + 1
      }
   }
   # Fix the names
   colnames(out) <- yU
   rownames(out) <- xU
   # Return the kriged value
   return(out)
}

### ====================================================
### =                    E N D                         =
### ====================================================
