library(quanteda) 

 

remove_overlap <- function(subdf){ 

    df_out <- subdf[1,] # First record - direct save  

    if(NROW(subdf)>1){ 

         

        # Go over remaining records - remove overlap 

        for(i in 2:NROW(subdf)){ 

            num_overlap <- subdf$right[i-1] - subdf$left[i] + 1 

            if(num_overlap>0){ # Need to deduplicate  

                j <- NROW(df_out) 

                new_words <- strsplit(subdf$context[i],"[[:blank:]]+") 

                new_words <- unlist(new_words) 

                N <- length(new_words) 

                num_keep <- N - num_overlap 

                if(num_keep>0){ 

                    keep_words <- new_words[(num_overlap+1):N] 

                    df_out$context[j] <- paste0(df_out$context[j], " ", 

                                                paste0(keep_words,  

                                                       collapse =" ")) 

                } 

            } else { 

                df_out <- rbind(df_out,subdf[i,]) 

            } 

        } 

    } 

    return(df_out) 

} 

 

extract_words <- function(kws, docs, size){ 

     

    # Tokenize textual data 

    docCorpus <- tokens(docs) 

    docLen <- sapply(docCorpus, length) 

    docLen_df <- data.frame(docname=names(docLen), 

                        doc_num_words=as.vector(docLen)) 

     

    # Capture words around of Innovation terms (kws) 

    result <- kwic(docCorpus, pattern=phrase(kws), window = size) 

     

    df <- as.data.frame(result) 

    df <- merge(df, docLen_df, by="docname", all.x = TRUE, all.y=FALSE) 

    df$left <- ifelse(df$from - size<1, 1, df$from - size) 

    df$right <- ifelse(df$to + size > df$doc_num_words,  

                       df$doc_num_words, df$to + size) 

    df$context <- paste(df$pre, df$keyword, df$post) 

    df <- df[,c("docname", "left", "right", "context")] 

     

    # Identify and fix overlap 

    uni_docs <- unique(df$docname) 

    df_out <- NULL # where we save output  

    for(d in uni_docs){ 

        subdf <- subset(df, docname==d) 

        subdf <- remove_overlap(subdf) 

        df_out <- rbind(df_out, subdf) 

    } 

     

    # aggregate to firm level 

    df_out_agg <- aggregate(df_out$context, list(df_out$docname), paste, collapse=" ") 

    colnames(df_out_agg) <- c("docname","text") 

    return(df_out_agg) 

} 

 

extract_units <- function(kws, units){ 

    result <- data.frame(docname=names(units), text="") 

     

    # go over each document 

    for(i in 1:length(units)){  

        doc_units <- units[[i]] 

        doc_tokens <- strsplit(doc_units, "\\s+") 

        match <- c() 

         

        # go over each unit (e.g., sent) 

        for(j in 1:length(doc_units)){ 

            doc_wds <- doc_tokens[[j]] 

            overlap <- intersect(doc_wds,kws) 

            if(length(overlap)>0){ 

                match <- c(match,doc_units[j]) 

            } 

        } 

        result$text[i] <- paste0(match, collapse = " ") 

    } 

    return(result) 

} 

 

extract_collocation <- function(kws, docs, level="word", size=4){ 

    if(level=="word"){ # find context by nearby words 

        colloc <- extract_words(kws, docs, size) 

    }  

    if(grepl("^sent",level)){ # find context by sentence 

        units <- tokens(docs, what = "sentence") 

        colloc <- extract_units(kws, units) 

    } 

    return(colloc) 

} 
