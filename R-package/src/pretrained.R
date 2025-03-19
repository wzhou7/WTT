# Finding Related Words Using Pretrained Word Vectors 

# load the C function to read Google's binary word vector file 
dyn.load("distance.dll") 

pretrained <- function(model_name, focal_terms, num_words){ 
    data <- NULL
    for(w in focal_terms){ 

        OUT <- .C("distance", 

                  rfile_name=as.character(model_name), 

                  search_word=as.character(w), 

                  rN=as.integer(num_words), 

                  rbestw=as.character(rep("123",num_words)), 

                  rbestd=as.double(rep(0,num_words))) 

         

        subdata <- data.frame(term1=w, 

                              term2=OUT$rbestw, 

                              similarity=OUT$rbestd, 

                              rank=1:num_words) 

         

        # filter out the non-vocabulary words 

        subdata <- subset(subdata,term2!="") 

        data <- rbind(data,subdata) 

    } 

    write.csv(data, file=paste0(model_name,"_results.csv"),  

              row.names = FALSE) 

} 
