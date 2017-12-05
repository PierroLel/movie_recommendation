## The purpose of this script is too compute user rating prediction (or recommendation if you will)
## based on a dataset from movielens.
## The priority nulber one is to learn how to set up some ML algorithlms and how to thune them right

################ Load the data: ################
ratings <- read_delim("C:/Users/woill/Documents/ml-100k/u1 test 1.txt", 
            "\t", escape_double = FALSE, col_names = c("u_id","i_id","rat","tstp"),
            col_types = cols(u_id = col_integer(),i_id=col_integer(),rat=col_integer(),tstp=col_integer())
            )
ratings=data.frame(ratings)

users <- read_delim("C:/Users/woill/Documents/ml-100k/user 1.txt","|", escape_double = FALSE, col_names = c("u_id","age","gender","occupation","zip_code")
                    ,col_types = cols(u_id = col_integer(),age=col_integer(),gender=col_character(),occupation=col_character(),zip_code=col_character())
            )
users=data.frame(users)

items <- read_delim("C:/Users/woill/Documents/ml-100k/u item 1.txt", 
            "|", escape_double = FALSE, col_names =c("i_id","title","release","video_release","imdb","unknown","action","adventure","animation"
                                                             ,"children","comedy","crime","documentary","drama","fantasy","film_noir","horror","musical"
                                                             ,"mystery","romance","sci_fi","thriller","war","western"))
items=data.frame(items)
items["rele_y"]=strtoi(substr(items$release,11-3,11))

#View(items)
################ Model 1: Collaborative Filtering ################
#### Step 1 ####
## user_id in [1:U], item_id in [1:M]
## y(i,j) = rating from user j for movie i
## y(i,j) = (theta(j))T * x(i)
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## creation of the rating mùatrix of size (M,U), with rows = item_id
## & col = user_1 ratings | user_2 ratings | user_3 ratings | .... | user_U ratings 
## and inside the actual rating from 1 to 5 (integerss)
## basically the Y matrix where y(i,j) = rating from user j for movie i

test=ratings#[1:1000,]

U=max(test["u_id"]); M=max(test["i_id"]) ## some are missing, for exampl movie_id goes from 153 to 157... that kind of thing
## So better to work with the indexes in this case
a=cast(ratings[,1:3],i_id~u_id)
b=data.frame(a,row.names = a[,1]) #,colnames=colnames(a[-1]))
Y=b

## let us note A the number of possible genres
## Creation of the movie attributes table:
## size (M,A) such as: rows= item_id and columns = genre_1 | genre_2 | ... | genre_A | release Year
## normalization of release_year:
X= items[,c(1,6:25)]
X["rele_y"]=(X$rele_y-mean(X$rele_y,na.rm = T))/sd(X$rele_y,na.rm = T)
X[is.na(X$rele_y),]=0

## Creation of the user attribute table (full of random value (0.1) for now)
size=dim(X)[2]*dim(users)[1]
b=matrix(0.1,nrow=dim(users)[1],ncol=length(X))
bb=data.frame(b); rownames(bb) = users$u_id; colnames(bb) = c("u_id",colnames(X[-1]))
bb["u_id"]=users$u_id
theta=bb

## make sure all indexes are here (to ensure the right dimension for the vectors)
colY=colnames(Y); indeY=Y$i_id; indeTh=theta$u_id;  indeX=X$i_id

nbM=intersect(strtoi(indeY),indeX)
nbU=intersect(strtoi(substr(colY,2,5)),indeTh)

v1=theta[theta$u_id %in% nbU,]; theta=v1
v2=X[X$i_id %in% nbM,];  X=v2
v3=Y[Y$i_id %in% nbM,c("i_id",paste("X",nbU,sep=""))]; Y=v3

#### Step 2 ####
## from X estimate theta: 
## for all u, do theta_u=arg(regressions(y_u~x)): (u are the individuals)

trn1=list()
prd1=list()
cnam=c("u_id",colnames(X[,2:dim(X)[2]]))
mat=matrix(0,nrow = length(unique(ratings$u_id)),ncol = length(cnam))
theta1=data.frame(mat,row.names = unique(ratings$u_id))
colnames(theta1)=cnam
theta1$u_id= unique(ratings$u_id)

for (u in 2:dim(Y)[2]) {
  #u=2 #reference of the user
  Y_u=Y[, c(1, u)] #get the movies' id and the rate u gave to them
  M_u=Y_u[!is.na(Y_u[2]), 1] #get the id of the movies u has seen
  y_u=Y_u[Y_u$i_id %in% M_u, ] #remove those he hasn't rated
  
  y_trn=y_u
  name_u=strtoi(substr(colnames(y_trn)[2],2,5)) 
  trn1[name_u]=list(y_trn)
  ## store both to do train Vs test comparaison later on
  
  xinde=intersect(M_u,y_trn$i_id)
  x=X[X$i_id %in% xinde,] ## get the according movies' genre
  colnames(y_trn)=c("i_id","y")
  y_prd=y_trn
  sampl=merge (y_trn, x, by = "i_id") ## join to make the regression easy
  regr_u=lm(y~0+.,data = as.data.frame(sampl[,-1]))#,weights = matrix(0.1,nrow = 1,ncol = (dim(sampl)[2]-2)))
  
  #store the theta_u
  theta_u=coef(regr_u)
  theta_u[is.na(theta_u)]=0
  theta1[theta1$u_id==name_u,]=cbind("u_id"=name_u,t(theta_u))
  cbind(y_trn,predict(regr_u))
  y_prd$y=predict(regr_u)
  prd1[name_u]=list(y_prd)
}

#### Step 3 ####
## from theta estimate X:
## for all m, do x_m=arg(regressions(y_m~theta)): (m are the movies)

tst1=list()
trn1=list()
prd1=list()
cnam=c("i_id",colnames(theta1[,2:dim(theta1)[2]]))
mat=matrix(0,nrow = length(unique(items$i_id)),ncol = length(cnam))
X1=data.frame(mat,row.names = unique(items$i_id))
colnames(X1)=cnam
X1$i_id= unique(items$i_id)


for (m in Y$i_id) {
  #m=5 #reference of the item/movie
  Y_m=Y[Y$i_id==m,] #get the users that rated m
  Y_m=Y_m[,-1]
  Y_m["u_id",]=strtoi(substr(colnames(Y_m),2,5))
  Y_m=t(Y_m); mm=paste("m",m,sep="")
  colnames(Y_m)=c(mm,"u_id")
  Y_m=as.data.frame((Y_m))
  
  U_m=Y_m[!is.na(Y_m[,mm]),"u_id"] #get the id of the movies u has seen
  y_m=Y_m[Y_m$u_id %in% U_m, ] #remove those he hasn't rated
  
  ## train test split ##
  y_trn=y_m
  name_m=strtoi(substr(colnames(y_trn)[1],2,5)) #=m
  trn1[name_m]=list(y_trn)
  ## store both to do train Vs test comparaison later on
  
  thinde=intersect(U_m,y_trn$u_id)
  th=theta1[theta1$u_id %in% thinde,] ## get the according movies' genre
  colnames(y_trn)=c("y","u_id")
  y_prd=y_trn
  sampl=merge (y_trn, th, by = "u_id") ## join to make the regression easy
  regr_m=lm(y~0+.,data = as.data.frame(sampl[,-1]))#,weights = matrix(0.1,nrow = 1,ncol = (dim(sampl)[2]-2)))
  
  #store the x_m
  x_m=coef(regr_m)
  x_m[is.na(x_m)]=0
  X1[X1$i_id==name_m,]=cbind("m_id"=name_m,t(x_m))
  cbind(y_trn,predict(regr_m))
  y_prd$y=predict(regr_m)
  prd1[name_m]=list(y_prd)
}


## 
#### Step 4 ####
## from X estimate Theta1; from Theta1 estimate X1; from X1 estimate Theta1.....

## inicialization :
#estimate theta from X
trn1=list()
prd1=list()
cnam=c("u_id",colnames(X[,2:dim(X)[2]]))
mat=matrix(0,nrow = length(unique(ratings$u_id)),ncol = length(cnam))
theta1=data.frame(mat,row.names = unique(ratings$u_id))
colnames(theta1)=cnam
theta1$u_id= unique(ratings$u_id)
for (u in 2:dim(Y)[2]) {
  #u=2 #reference of the user
  Y_u=Y[, c(1, u)] #get the movies' id and the rate u gave to them
  M_u=Y_u[!is.na(Y_u[2]), 1] #get the id of the movies u has seen
  y_u=Y_u[Y_u$i_id %in% M_u, ] #remove those he hasn't rated
  
  y_trn=y_u
  name_u=strtoi(substr(colnames(y_trn)[2],2,5)) 
  trn1[name_u]=list(y_trn)
  ## store both to do train Vs test comparaison later on
  
  xinde=intersect(M_u,y_trn$i_id)
  x=X[X$i_id %in% xinde,] ## get the according movies' genre
  colnames(y_trn)=c("i_id","y")
  y_prd=y_trn
  sampl=merge (y_trn, x, by = "i_id") ## join to make the regression easy
  regr_u=lm(y~0+.,data = as.data.frame(sampl[,-1]))#,weights = matrix(0.1,nrow = 1,ncol = (dim(sampl)[2]-2)))
  
  #store the theta_u
  theta_u=coef(regr_u)
  theta_u[is.na(theta_u)]=0
  theta1[theta1$u_id==name_u,]=cbind("u_id"=name_u,t(theta_u))
  cbind(y_trn,predict(regr_u))
  y_prd$y=predict(regr_u)
  prd1[name_u]=list(y_prd)
}

y_compar=matrix(ncol = 3)
y_comp1=data.frame(y_compar)
colnames(y_comp1)=c("i_id","trn","prd")
y_comp1=y_comp1[-1,]
for (i in strtoi(substr(colnames(Y)[-1],2,5))) {
  y_tr=trn1[[i]]
  y_pr=prd1[[i]]
  y_comp=merge(y_tr,y_pr,by="i_id"); colnames(y_comp)=colnames(y_comp1)
  y_comp1=rbind(y_comp1,y_comp)
}
y_comp1["diff"]=y_comp1$trn-y_comp1$prd
y_comp1["diff2"]=y_comp1$diff**2
hist(y_comp1["diff"],breaks = 35)
title(main = "inicialization")
print("inicialization :")
print(mean(y_comp1$diff2))


## start the collaborative:
## 
#estimate X from theta
k=1
while (k<=30) {
  
  tst1=list()
  trn1=list()
  prd1=list()
  cnam=c("i_id",colnames(theta1[,2:dim(theta1)[2]]))
  mat=matrix(0,nrow = length(unique(items$i_id)),ncol = length(cnam))
  X1=data.frame(mat,row.names = unique(items$i_id))
  colnames(X1)=cnam
  X1$i_id= unique(items$i_id)
  for (m in Y$i_id) {
    #m=5 #reference of the item/movie
    Y_m=Y[Y$i_id==m,] #get the users that rated m
    Y_m=Y_m[,-1]
    Y_m["u_id",]=strtoi(substr(colnames(Y_m),2,5))
    Y_m=t(Y_m); mm=paste("m",m,sep="")
    colnames(Y_m)=c(mm,"u_id")
    Y_m=as.data.frame((Y_m))
    
    U_m=Y_m[!is.na(Y_m[,mm]),"u_id"] #get the id of the movies u has seen
    y_m=Y_m[Y_m$u_id %in% U_m, ] #remove those he hasn't rated
    
    ## train test split ##
    y_trn=y_m
    name_m=strtoi(substr(colnames(y_trn)[1],2,5)) #=m
    trn1[name_m]=list(y_trn)
    ## store both to do train Vs test comparaison later on
    
    thinde=intersect(U_m,y_trn$u_id)
    th=theta1[theta1$u_id %in% thinde,] ## get the according movies' genre
    colnames(y_trn)=c("y","u_id")
    y_prd=y_trn
    sampl=merge (y_trn, th, by = "u_id") ## join to make the regression easy
    regr_m=lm(y~0+.,data = as.data.frame(sampl[,-1]))#,weights = matrix(0.1,nrow = 1,ncol = (dim(sampl)[2]-2)))
    
    #store the x_m
    x_m=coef(regr_m)
    x_m[is.na(x_m)]=0
    X1[X1$i_id==name_m,]=cbind("m_id"=name_m,t(x_m))
    cbind(y_trn,predict(regr_m))
    y_prd$y=predict(regr_m)
    prd1[name_m]=list(y_prd)
  }
  
  y_compar=matrix(ncol = 3)
  y_comp1=data.frame(y_compar)
  colnames(y_comp1)=c("u_id","trn","prd")
  y_comp1=y_comp1[-1,]
  for (i in strtoi(row.names(Y)[-1])) { #strtoi(substr(colnames(Y)[-1],2,5))) {
    y_tr=trn1[[i]]
    y_pr=prd1[[i]]
    y_comp=merge(y_tr,y_pr,by="u_id"); colnames(y_comp)=colnames(y_comp1)
    y_comp1=rbind(y_comp1,y_comp)
  }
  y_comp1["diff"]=y_comp1$trn-y_comp1$prd
  y_comp1["diff2"]=y_comp1$diff**2
  hist(y_comp1["diff"],breaks = 35)
  title(main = paste(" X from th iteration no",k))
  print(paste("X from th iteration no",k,": ",mean(y_comp1$diff2)))
  
  
  
  #estimate theta from X
  trn1=list()
  prd1=list()
  cnam=c("u_id",colnames(X1[,2:dim(X1)[2]]))
  mat=matrix(0,nrow = length(unique(ratings$u_id)),ncol = length(cnam))
  theta1=data.frame(mat,row.names = unique(ratings$u_id))
  colnames(theta1)=cnam
  theta1$u_id= unique(ratings$u_id)
  for (u in 2:dim(Y)[2]) {
    #u=2 #reference of the user
    Y_u=Y[, c(1, u)] #get the movies' id and the rate u gave to them
    M_u=Y_u[!is.na(Y_u[2]), 1] #get the id of the movies u has seen
    y_u=Y_u[Y_u$i_id %in% M_u, ] #remove those he hasn't rated
    
    y_trn=y_u
    name_u=strtoi(substr(colnames(y_trn)[2],2,5)) 
    trn1[name_u]=list(y_trn)
    ## store both to do train Vs test comparaison later on
    
    xinde=intersect(M_u,y_trn$i_id)
    x=X1[X1$i_id %in% xinde,] ## get the according movies' genre
    colnames(y_trn)=c("i_id","y")
    y_prd=y_trn
    sampl=merge (y_trn, x, by = "i_id") ## join to make the regression easy
    regr_u=lm(y~0+.,data = as.data.frame(sampl[,-1]))#,weights = matrix(0.1,nrow = 1,ncol = (dim(sampl)[2]-2)))
    
    #store the theta_u
    theta_u=coef(regr_u)
    theta_u[is.na(theta_u)]=0
    theta1[theta1$u_id==name_u,]=cbind("u_id"=name_u,t(theta_u))
    cbind(y_trn,predict(regr_u))
    y_prd$y=predict(regr_u)
    prd1[name_u]=list(y_prd)
  }
  
  y_compar=matrix(ncol = 3)
  y_comp1=data.frame(y_compar)
  colnames(y_comp1)=c("i_id","trn","prd")
  y_comp1=y_comp1[-1,]
  for (i in strtoi(substr(colnames(Y)[-1],2,5))) {
    y_tr=trn1[[i]]
    y_pr=prd1[[i]]
    y_comp=merge(y_tr,y_pr,by="i_id"); colnames(y_comp)=colnames(y_comp1)
    y_comp1=rbind(y_comp1,y_comp)
  }
  y_comp1["diff"]=y_comp1$trn-y_comp1$prd
  y_comp1["diff2"]=y_comp1$diff**2
  hist(y_comp1["diff"],breaks = 35)
  title(main = paste(" th from X iteration no",k))
  print(paste(" th from X iteration no",k,": ",mean(y_comp1$diff2)))
  k=k+1
}


#### Step 5 ####
## train test split:
Yall=Y
Ytst=Yall;Ytrn=Yall
iii=0
for (u in 2:dim(Yall)[2]) {
  #u=2
  Y_u=Yall[, c(1, u)] #get the movies' id and the rate u gave to them
  M_u=Y_u[!is.na(Y_u[2]), 1] #get the id of the movies u has seen
  y_u=Y_u[Y_u$i_id %in% M_u, ] #remove those he hasn't rated
  for (m in y_u$i_id) {
    #m=6
    if (runif(1,0,1) < 0.2) {
      Ytrn[Ytrn$i_id==m,u]=NA
      
    } else {
      Ytst[Ytst$i_id==m,u]=NA
    }
    i=i+1
  }
}

#### Step 6 ####
## compare Train to Test
## inicialization :
#estimate theta from X

Y=Ytrn
trn1=list()
tst1=list()
prd1=list()
prd_tst1=list()
cnam=c("u_id",colnames(X[,2:dim(X)[2]]))
mat=matrix(0,nrow = length(unique(ratings$u_id)),ncol = length(cnam))
theta1=data.frame(mat,row.names = unique(ratings$u_id))
colnames(theta1)=cnam
theta1$u_id= unique(ratings$u_id)
for (u in 2:dim(Y)[2]) {
  #u=2 #reference of the user
  Y_u=Y[, c(1, u)] #get the movies' id and the rate u gave to them
  M_u=Y_u[!is.na(Y_u[2]), 1] #get the id of the movies u has seen
  y_u=Y_u[Y_u$i_id %in% M_u, ] #remove those he hasn't rated
  
  Y_u_tst=Ytst[, c(1, u)] #get the movies' id and the rate u gave to them
  M_u_tst=Y_u_tst[!is.na(Y_u_tst[2]), 1] #get the id of the movies u has seen
  y_u_tst=Y_u_tst[Y_u_tst$i_id %in% M_u_tst, ] #remove those he hasn't rated
  
  
  
  y_trn=y_u
  y_tst=y_u_tst
  name_u=strtoi(substr(colnames(y_trn)[2],2,5)) 
  trn1[name_u]=list(y_trn)
  tst1[name_u]=list(y_tst)
  ## store both to do train Vs test comparaison later on
  
  xinde=intersect(M_u,y_trn$i_id)
  xinde_tst=intersect(M_u_tst,y_tst$i_id)
  x=X[X$i_id %in% xinde,] ## get the according movies' genre
  x_tst=X[X$i_id %in% xinde_tst,]
  colnames(y_trn)=c("i_id","y")
  colnames(y_tst)=c("i_id","y")
  y_prd=y_trn
  y_prd_tst=y_tst
  sampl=merge (y_trn, x, by = "i_id") ## join to make the regression easy
  regr_u=lm(y~0+.,data = as.data.frame(sampl[,-1]))#,weights = matrix(0.1,nrow = 1,ncol = (dim(sampl)[2]-2)))
  
  #store the theta_u
  theta_u=coef(regr_u)
  theta_u[is.na(theta_u)]=0
  theta1[theta1$u_id==name_u,]=cbind("u_id"=name_u,t(theta_u))
  cbind(y_trn,predict(regr_u))
  y_prd$y=predict(regr_u)
  y_prd_tst$y= as.matrix(x_tst[,-1]) %*% as.matrix(theta_u)
  prd1[name_u]=list(y_prd)
  prd_tst1[name_u]=list(y_prd_tst)
}

y_compar=matrix(ncol = 3)
y_comp1=data.frame(y_compar)
colnames(y_comp1)=c("i_id","trn","prd")
y_comp1=y_comp1[-1,]
for (i in strtoi(substr(colnames(Y)[-1],2,5))) {
  y_tr=trn1[[i]]
  y_pr=prd1[[i]]
  y_comp=merge(y_tr,y_pr,by="i_id"); colnames(y_comp)=colnames(y_comp1)
  y_comp1=rbind(y_comp1,y_comp)
}
y_comp_tst1["diff"]=y_comp1$trn-y_comp1$prd
y_comp1["diff2"]=y_comp1$diff**2
y_compar=matrix(ncol = 3)
y_comp_tst1=data.frame(y_compar)
colnames(y_comp_tst1)=c("i_id","trn","prd")
y_comp_tst1=y_comp_tst1[-1,]
for (i in strtoi(substr(colnames(Y)[-1],2,5))) {
  y_tr=tst1[[i]]
  y_pr=prd_tst1[[i]]
  y_comp=merge(y_tr,y_pr,by="i_id"); colnames(y_comp)=colnames(y_comp_tst1)
  y_comp_tst1=rbind(y_comp_tst1,y_comp)
}
y_comp_tst1["diff"]=y_comp_tst1$trn-y_comp_tst1$prd
y_comp_tst1["diff2"]=y_comp_tst1$diff**2
hist(y_comp_tst1["diff"],breaks = 35)
title(main = "inicialization")
print("inicialization :")
print(mean(y_comp_tst1$diff2))


## start the collaborative:
## 
#estimate X from theta
k=1
while (k<=30) {
  
  tst1=list()
  trn1=list()
  prd1=list()
  prd_tst1=list()
  cnam=c("i_id",colnames(theta1[,2:dim(theta1)[2]]))
  mat=matrix(0,nrow = length(unique(items$i_id)),ncol = length(cnam))
  X1=data.frame(mat,row.names = unique(items$i_id))
  colnames(X1)=cnam
  X1$i_id= unique(items$i_id)
  for (m in Y$i_id) {
    #m=1 #reference of the item/movie
    if (sum(Y[Y$i_id==m,-1],na.rm = T)!=0) {
      Y_m=Y[Y$i_id==m,] #get the users that rated m
      Y_m=Y_m[,-1]
      Y_m["u_id",]=strtoi(substr(colnames(Y_m),2,5))
      Y_m=t(Y_m); mm=paste("m",m,sep="")
      colnames(Y_m)=c(mm,"u_id")
      Y_m=as.data.frame((Y_m))
      
        if (sum(Ytst[Ytst$i_id==m,-1],na.rm = T)!=0) {
        Y_m_tst=Ytst[Ytst$i_id==m,] #get the users that rated m
        Y_m_tst=Y_m_tst[,-1]
        Y_m_tst["u_id",]=strtoi(substr(colnames(Y_m_tst),2,5))
        Y_m_tst=t(Y_m_tst); mm=paste("m",m,sep="")
        colnames(Y_m_tst)=c(mm,"u_id")
        Y_m_tst=as.data.frame((Y_m_tst))
        
        U_m=Y_m[!is.na(Y_m[,mm]),"u_id"] #get the id of the movies u has seen
        y_m=Y_m[Y_m$u_id %in% U_m, ] #remove those he hasn't rated
        U_m_tst=Y_m_tst[!is.na(Y_m_tst[,mm]),"u_id"] #get the id of the movies u has seen
        y_m_tst=Y_m_tst[Y_m_tst$u_id %in% U_m_tst, ] #remove those he hasn't rated
        
        ## train test split ##
        y_trn=y_m
        y_tst=y_m_tst
        name_m=strtoi(substr(colnames(y_trn)[1],2,5)) #=m
        trn1[name_m]=list(y_trn)
        tst1[name_m]=list(y_tst)
        ## store both to do train Vs test comparaison later on
        
        thinde=intersect(U_m,y_trn$u_id)
        thinde_tst=intersect(U_m_tst,y_tst$u_id)
        th=theta1[theta1$u_id %in% thinde,] ## get the according movies' genre
        th_tst=theta1[theta1$u_id %in% thinde_tst,]
        colnames(y_trn)=c("y","u_id")
        colnames(y_tst)=c("y","u_id")
        y_prd=y_trn
        y_prd_tst=y_tst
        sampl=merge (y_trn, th, by = "u_id") ## join to make the regression easy
        
        
        
        if (length(unique(sampl[,2]))==1) {
          regr_m=lm(y~0+.,data = as.data.frame(sampl[,-1]))
          x_m=coef(regr_m)
        } else {
          regr_m=glmnet(x=as.matrix(sampl[,3:22]),y=as.matrix(sampl[,2]),lambda = c(0.001),intercept = F)
          x_m=coef(regr_m)[-1,]
        }
        
        #,weights = matrix(0.1,nrow = 1,ncol = (dim(sampl)[2]-2)))
        
        #cbind(sampl[,2],predict(regr_m, newx = as.matrix(sampl[,3:22])))
        
        #store the x_m
        #x_m=coef(regr_m)
        #x_m1=coef(regr_m1)
        x_m[is.na(x_m)]=0
        X1[X1$i_id==name_m,]=cbind("m_id"=name_m,t(x_m))
        #cbind(y_trn,predict(regr_m))
        y_prd$y=predict(regr_m , newx = as.matrix(sampl[,3:22]))
        y_prd_tst$y= as.matrix(th_tst[,-1]) %*% as.matrix(x_m)
        prd1[name_m]=list(y_prd)
        prd_tst1[name_m]=list(y_prd_tst)
      }
    }
  }
  
  y_compar=matrix(ncol = 3)
  y_comp1=data.frame(y_compar)
  colnames(y_comp1)=c("u_id","trn","prd")
  y_comp1=y_comp1[-1,]
  for (i in strtoi(row.names(Y)[-1])) { #strtoi(substr(colnames(Y)[-1],2,5))) {
    if (i <length(trn1)) {
    y_tr=trn1[[i]]
    y_pr=prd1[[i]]
      if ((sum(y_pr,na.rm = T)!=0)&&(sum(y_tr,na.rm = T)!=0)) {
        y_comp=merge(y_tr,y_pr,by="u_id"); colnames(y_comp)=colnames(y_comp1)
        y_comp1=rbind(y_comp1,y_comp)
      }
    }
  }
  y_comp1["diff"]=y_comp1$trn-y_comp1$prd
  y_comp1["diff2"]=y_comp1$diff**2
  
  y_compar_tst=matrix(ncol = 3)
  y_comp1_tst=data.frame(y_compar)
  colnames(y_comp1_tst)=c("u_id","trn","prd")
  y_comp1_tst=y_comp1_tst[-1,]
  for (i in strtoi(row.names(Y)[-1])) { #strtoi(substr(colnames(Y)[-1],2,5))) {
    if (i <length(tst1)) {
      y_tr=tst1[[i]]
      y_pr=prd_tst1[[i]]
      if ((dim(y_tr)[1]!=0)&&(sum(y_pr,na.rm = T)!=0)&&(sum(y_tr,na.rm = T)!=0)) {
        y_comp=merge(y_tr,y_pr,by="u_id"); colnames(y_comp)=colnames(y_comp1_tst)
        y_comp1_tst=rbind(y_comp1_tst,y_comp)
      }
    }
  }
  y_comp1_tst["diff"]=y_comp1_tst$trn-y_comp1_tst$prd
  y_comp1_tst["diff2"]=y_comp1_tst$diff**2
  
  
  par(bg="cornsilk",lwd=2,col="black",mfrow=c(1,2))
  hist(y_comp1["diff"],breaks = 35)
  title(main = paste(" X from th iteration no",k))
  print(paste("X from th iteration no",k,": ",mean(y_comp1$diff2)))
  hist(y_comp1_tst["diff"],breaks = 35)
  title(main = paste(" X from th iteration no",k))
  print(paste("X from th iteration no",k,": ",mean(y_comp1_tst$diff2)))
  
  
  
  
  
  
  
  #estimate theta from X
  trn1=list()
  tst1=list()
  prd1=list()
  prd_tst1=list()
  cnam=c("u_id",colnames(X1[,2:dim(X1)[2]]))
  mat=matrix(0,nrow = length(unique(ratings$u_id)),ncol = length(cnam))
  theta1=data.frame(mat,row.names = unique(ratings$u_id))
  colnames(theta1)=cnam
  theta1$u_id= unique(ratings$u_id)
  for (u in 2:dim(Y)[2]) {
    #u=2 #reference of the user
    if (sum(Y[,u],na.rm=T)!=0) {
      Y_u=Y[, c(1, u)] #get the movies' id and the rate u gave to them
      M_u=Y_u[!is.na(Y_u[2]), 1] #get the id of the movies u has seen
      y_u=Y_u[Y_u$i_id %in% M_u, ] #remove those he hasn't rated
      
      y_trn=y_u
      name_u=strtoi(substr(colnames(y_trn)[2],2,5)) 
      trn1[name_u]=list(y_trn)
      ## store both to do train Vs test comparaison later on
      
      xinde=intersect(M_u,y_trn$i_id)
      x=X1[X1$i_id %in% xinde,] ## get the according movies' genre
      if (sum(x[,2:21])!=0) {
      colnames(y_trn)=c("i_id","y")
      y_prd=y_trn
      sampl=merge (y_trn, x, by = "i_id") ## join to make the regression easy
      
      if (length(unique(sampl[,2]))==1) {
        regr_u=lm(y~0+.,data = as.data.frame(sampl[,-1]))
        theta_u=coef(regr_u)
      } else {
        regr_u=glmnet(x=as.matrix(sampl[,3:22]),y=as.matrix(sampl[,2]),lambda = c(0.001),intercept = F)
        theta_u=coef(regr_u)[-1,]
      }
      
      
      #regr_u=lm(y~0+.,data = as.data.frame(sampl[,-1]))#,weights = matrix(0.1,nrow = 1,ncol = (dim(sampl)[2]-2)))
      
      #store the theta_u
      
      theta_u[is.na(theta_u)]=0
      theta1[theta1$u_id==name_u,]=cbind("u_id"=name_u,t(theta_u))
      
      y_prd$y=predict(regr_u , newx = as.matrix(sampl[,3:22]))
      prd1[name_u]=list(y_prd)
      }
    }
  }
  
  y_compar=matrix(ncol = 3)
  y_comp1=data.frame(y_compar)
  colnames(y_comp1)=c("i_id","trn","prd")
  y_comp1=y_comp1[-1,]
  for (i in strtoi(substr(colnames(Y)[-1],2,5))) {
    y_tr=trn1[[i]]
    y_pr=prd1[[i]]
    if ((dim(y_tr)[1]!=0)&&(sum(y_pr,na.rm = T)!=0)&&(sum(y_tr,na.rm = T)!=0)) {
      y_comp=merge(y_tr,y_pr,by="i_id"); colnames(y_comp)=colnames(y_comp1)
      y_comp1=rbind(y_comp1,y_comp)
    }
  }
  y_comp1["diff"]=y_comp1$trn-y_comp1$prd
  y_comp1["diff2"]=y_comp1$diff**2
  hist(y_comp1["diff"],breaks = 35)
  title(main = paste(" th from X iteration no",k))
  print(paste(" th from X iteration no",k,": ",mean(y_comp1$diff2)))
  k=k+1
}




## 




## 
## 
## 
## 
## 
dim(X1)

xxx=merge(X,X1, by = "i_id")
for (i in 2:21) {
  xxx[paste(colnames(xxx)[i],"diff")]=xxx[,i]-xxx[,i+20]
}
for (i in 2:21) {
  xxx[paste(colnames(xxx)[i],"diff2")]=xxx[paste(colnames(xxx)[i],"diff")]**2
}
par(bg="cornsilk",lwd=2,col="black",mfrow=c(1,2))
for (i in 2:21) {
  plot(xxx[,i],xxx[,i+20])
  title(paste(colnames(xxx)[i]))
  hist(xxx[paste(colnames(xxx)[i],"diff")])
  title(paste(colnames(xxx)[i]))
}
for (i in 42:61) {
  print(colnames(xxx)[i])
  hist(xxx[,i],breaks = 35)
} 
for (i in xxx$i_id) {
  xxx[xxx$i_id==i,"diffall"]=mean(as.matrix(xxx[xxx$i_id==i,62:81]))
}

mean(xxx$diffall)



##
y_comp1_tst["prd2"]=round(y_comp1_tst$prd,digits = 0)
y_comp1_tst["dif"]=y_comp1_tst$trn-y_comp1_tst$prd2
df=data.frame(count=c(0,0,0,0,0),ecart = c(-2:2))
for (i in -1:1){
  df[df$ecart==i,"count"]=dim(y_comp1_tst[y_comp1_tst$dif==i,])[1]
}
df[df$ecart==2,"count"]=dim(y_comp1_tst[y_comp1_tst$dif>=2,])[1]
df[df$ecart==-2,"count"]=dim(y_comp1_tst[y_comp1_tst$dif<=-2,])[1]

df["pct"]=100*df$count/sum(df$count)


##
y_compar=matrix(ncol = 3)
y_comp1=data.frame(y_compar)
colnames(y_comp1)=c("u_id","trn","prd")
y_comp1=y_comp1[-1,]
for (i in strtoi(row.names(Y)[-1])) { #strtoi(substr(colnames(Y)[-1],2,5))) {
  y_tr=trn1[[i]]
  y_pr=prd1[[i]]
  y_comp=merge(y_tr,y_pr,by="u_id"); colnames(y_comp)=colnames(y_comp1)
  y_comp1=rbind(y_comp1,y_comp)
}

par(bg="cornsilk",lwd=2,col="black")
y_comp1["diff"]=y_comp1$trn-y_comp1$prd
y_comp1["dif"]=y_comp1$trn-y_comp1$prd2
summary(y_comp1$dif);

cont = cast(y_comp1[,c("i_id","dif")] ,  i_id~dif, mean)

df=data.frame(count=c(0,0,0,0,0),ecart = c(-2:2))
for (i in -2:2){
  df[df$ecart==i,"count"]=dim(y_comp1[y_comp1$dif==i,])[1]
}
df["pct"]=100*df$count/sum(df$count)

sum()

y_comp1$dif=y_comp1$trn-y_comp1$prd2
hist(y_comp1$dif,breaks=500)
y_comp1["diff2"]=y_comp1$diff**2
#y_comp1$diff2=y_comp1$diff2/max(y_comp1$diff2)
y_comp1$diff2=abs(y_comp1$diff)%/%y_comp1$trn
y_comp1=transform(y_comp1,diff2=abs(diff)/trn)

hist(y_comp1$diff,breaks = 50,freq = F,col="orange")
curve(dnorm(x,mean=mean(y_comp1$diff),sd=sd(y_comp1$diff)),add=T,col="blue")
es=mean(y_comp1$diff2); va=sd(y_comp1$diff2)
al=((es**2)*(1-es)+es*va)/va; be=al*(1-es)/es
curve(dbeta(x,shape1 = al,shape2 = be),add=T,col="blue")

sum((y_comp1$tst-y_comp1$prd)**2)/dim(y_comp1)[1]
plot(x=y_comp1$tst-y_comp1$prd)

y_comp2=aggregate(c(y_comp1[,c(4,5)]), list(y_comp1$i_id),FUN=mean)
colnames(y_comp2)=c("i_id","m_diff","m2_diff")
hist(y_comp2$m2_diff,breaks = 100,freq = F,col="orange")
curve(dnorm(x,mean=mean(y_comp2$m2_diff),sd=sd(y_comp2$m2_diff)),add=T,col="blue")
y_comp2$m2_diff=(y_comp2$m2_diff-mean(y_comp2$m2_diff))/sd(y_comp2$m2_diff)
es=mean(y_comp2$m2_diff); va=sd(y_comp2$m2_diff)
al=((es**2)*(1-es)+es*va)/va; be=al*(1-es)/es
curve(dbeta(x,shape1 = al,shape2 = be),add=T,col="blue")



plot(regr_u)


?lm


########################

v=c()
for (i in 1:dim(x)[2]) {
  v=append(v,sum(x[,i]))
}
df=data.frame(t(v))
colnames(df)=colnames(x)

df1=data.frame(a=c(1,2,3,4),b=c(2,3,4,5),row.names = c(1,3,2,4))
df2=data.frame(d=c(1,2,3,4),e=c(2,3,4,5),row.names = c(1,2,3,4))

merge(df1, df2, by = 0)
?merge

n = c(2, 3, 5) 
s = c("aa", "bb", "cc", "dd", "ee") 
b = c(TRUE, FALSE, TRUE, FALSE, FALSE) 
x = list(a=n, c=s, d=b, 3)   # x contains copies of n, s, b
x$a
thet=rbind(theta1,cbind("u_id"=name_u,t(theta_u)))
rnam=row.names(thet)
theta1=data.frame(thet,row.names = rnam)
theta1[name_u,]=theta_u


########################





## J(theta) = cost function = sum over u of J_u(theta_u)
## J_u(theta_u) = cost function for user u; h(theta_u,xm)= approx for rating of movie m from user u
h1 = function(theta_u,xm) {
  tt1=array(   unlist(theta_u),  dim = c( nrow(theta_u), ncol(theta_u) ))
  tt2=array(   unlist(xm),  dim = c( nrow(xm), ncol(xm) ))
  return( tt1 %*% tt2 )
}

J_u=function(y_u,X,theta_u,h,lambda=0) {
  if ((sum(row.names(X)==row.names(Y))/dim(Y)[1])!=1) {
    return("check data index mthfck")
  }
  cst_M= t(y_u) - h(theta_u,t(X))
  cst_M[is.na(cst_M)]=0
  nrma=norm(cst_M,"2");normth=norm(theta_u,"2") ##
  res=(1/2)*nrma-(lambda/2)*normth
  return(res)
}

dJ_u = function(y_u,X,theta_u,h,lambda=0) {
  cst_M=h(X,t(theta_u))-y_u
  A=t(h1(cst_M,z_m))
  v=c()
  for (i in c(1:dim(A)[1]) ) {
    v=append(v,sum(t(h1(cst_M,z_m))[i,],na.rm = T))
    #print(sum(t(h1(cst_M,z_m))[i,],na.rm = T))
  }

}


cst_M=h1(z,t(tet))-g_u
A=t(h1(t(cst_M),z))
v=c()
for (i in c(1:dim(A)[1]) ) {
  v=append(v,sum(t(h1(cst_M,z_m))[i,],na.rm = T))
  #print(sum(t(h1(cst_M,z_m))[i,],na.rm = T))
}





?norm




J_u(g_u,z,tet,h1)

cst_M= t(g_u) - h1(tet,t(z))
cst_M[is.na(cst_M)]=0
nrma=norm(cst_M,"2")
normth=norm(tet,"2") ##
res=(1/2)*nrma-(lambda/2)*normth

sum(X$i_id==Y$i_id)/dim(Y)[1]

sum(theta$u_id==colnames(Y))/length(theta)

tet=theta[1,-1]
zm=X[1,-1]
z=X[,-1]
z_m=X[1,-1]
g_u=Y[,2]
h1(tet,t(zm))
vv=h1(tet,t(z))



g_u[is.na(g_u)]=0
z[is.na(z)]=0
vv=g_u-h1(tet,t(z));  vv[is.na(vv)]=0
h1(vv,z)
sum(h1(vv,z))
## dérivé de theta_u:
tet=theta[1,-1]j
z=X[,-1]
g_u=Y[,1]
g_u[is.na(g_u)]=0
z[is.na(z)]=0
vv=g_u-h1(tet,t(z));  vv[is.na(vv)]=0
alpha=0.1;lambda=0.1
sum(alpha*(h1(vv,z))+lambda*tet)








length(g_u[is.na(g_u)])
dim(g_u-h1(tet,t(z)))
dim(t(z))



dim(t(g_u-h1(tet,t(z))))
dim(z)

v1=c(1,2,3,4,5);v1
v2=c(1,3,2,4,7);v2

df1=data.frame(id=c(1,1,2,2),time=c(1,2,1,2), "4"=c(5,3,6,2), "6"=c(6,5,1,4),row.names = c(1,3,4,6))
df2=-data.frame(x1=c(1,1,2,2),x2=c(1,2,1,2), x3=c(5,3,6,2), x4=c(6,5,1,4),row.names = c(1,2,4,5))
v1=row.names(df1)
v2=colnames(df2)
vv=intersect(strtoi(substr(v2,2,5)), v1)
df2[df1$id %in% vv,paste("X",vv,sep="")]

vv;v2




################ Brouillon ################
strtoi(str_sub(v,-4,-1))

v=t(z)
tt1=array(   unlist(tet),  dim = c( nrow(tet[1]), ncol(tet) ))
tt2=array(   unlist(v),  dim = c( nrow(v[,1]), length(v[1,1])))
dim(tt1);dim(tt2);dim(tt1%*%tt2)

str = items["release"]
n = 4
substr(str,(nchar(str)+1)-n,nchar(str))

x <- items["release"]
substr(x, nchar(x)-n+1, nchar(x))
substr(x)

substr(items$release,1,4)
substr(items$release,11-3,11)

for(i in c(1:4)) {
  print(i)
}

if(1==1) {
  v="tr"
}


vv=matrix(zm)
v%*% vv
crossprod(array(tet),array(zm))

L=zm;v=array(unlist(L), dim = c(nrow(L[[1]]), ncol(L[[1]]), length(L)))
L=tet;vv=array(unlist(L), dim = c(nrow(L[[1]]), ncol(L[[1]]), length(L)))
data.frame(mapply(`*`,t(tet),zm))

a=matrix(c(1,2,3),ncol=3)
print (a %*% t(a))



x <- 1:4
(z <- x %*% x)

df <- data.frame(a=rnorm(100),b1=rnorm(100),b2=rnorm(100))
df[,fmatch(paste("b","1",sep = ""),names(df))]

df[,fmatch(paste("b",as.character(tet),sep = ""),names(df))]

max(strtoi(row.names(Y)))

dim(tet)

tt2=array(   unlist(zm),  dim = c( nrow(zm), ncol(zm) ))


df= data.frame(id=c(1,1,2,2),time=c(1,2,1,2), "4"=c(5,3,6,2), "6"=c(6,5,1,4),row.names = c(1,10,4,5))
v=c(1,3,2,99,88,10)
df[v,]
row



mydata= data.frame(id=c(1,1,2,2),time=c(1,2,4,3), "4"=c(5,3,6,2), "6"=c(6,5,1,4),row.names = c(1,2,4,5))
mdata <- melt(mydata, id=c("id","time"));mdata

mdata= data.frame(id=c(21:40),time=c(1:19,32), value=3)#,row.names = c(1,2,4,5))
subjmeans=cast(mdata, id~time);subjmeans=data.frame(subjmeans);subjmeans
subjmeans[,c(paste("x",c(1,3,19)),sep="")]


swiss <- datasets::swiss
x <- model.matrix(Fertility~., swiss)[,-1]
y <- swiss$Fertility
set.seed(489)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
ytest = y[test]
swisslm <- lm(Fertility~., data = swiss)
coef(swisslm)
ridge.mod <- glmnet(x, y, alpha = 0, lambda = lambda)

swisslm <- lm(Fertility~., data = swiss, subset = train)
ridge.mod <- glmnet(x[train,], y[train], alpha = 0, lambda = lambda)
#find the best lambda from our list via cross-validation
cv.out <- cv.glmnet(x[train,], y[train], alpha = 0)











