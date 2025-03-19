custrained <- function(model_name, focal_terms, num_words){ 

     

    # filter out non-existent words 

    model <- read.word2vec(model_name) 

    emb <- predict(model, focal_terms, type = "embedding") 

    term_exists <- apply(emb,1,function(x){sum(is.na(x))}==0) 

    focal_terms0 <- rownames(emb)[!term_exists] 

    print(paste0("These terms not found in the vocabulary: ", 

                 paste(focal_terms0, collapse=", "))) 

     

    # search for words 

    focal_terms1 <- rownames(emb)[term_exists] 

    data <- NULL 

    for(w in focal_terms1){         

        sim_words <- predict(model, w, type = "nearest",  

                             top_n = num_words)         

        subdata <- sim_words[[1]] 

        data <- rbind(data,subdata) 

    } 

    write.csv(data, paste0(model_name, "_results.csv"),  

              row.names = FALSE) 

} 
