require(kernlab)

triplet_similarity<-function(x,y){
    sk<-stringdot(type="spectrum", length=3, normalized=TRUE)
    m<-kernelMatrix(sk, x, y)
    return(m)
}