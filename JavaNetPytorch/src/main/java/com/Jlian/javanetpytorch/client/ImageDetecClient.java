package com.Jlian.javanetpytorch.client;

import org.springframework.util.MultiValueMap;

/**
    * Client层：用于和后面的深度学习框架进行同步的Http交互
    * @author JLian
    * @date 2021/11/16 9:54 下午
*/

public interface ImageDetecClient {

    /**
     * get detect image post
        * @param paramMap
        * @return java.lang.String
        * @author JLian
        * @date 2021/11/16 9:55 下午
    */
    String getDetectImagePost(MultiValueMap<String, String> paramMap);
}
