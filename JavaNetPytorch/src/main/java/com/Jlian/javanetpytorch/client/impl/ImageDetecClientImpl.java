package com.Jlian.javanetpytorch.client.impl;

import com.Jlian.javanetpytorch.client.ImageDetecClient;
import com.Jlian.javanetpytorch.constant.ViewModelConstants;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.util.Assert;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

@Component
public class ImageDetecClientImpl implements ImageDetecClient {
    @Override
    public String getDetectImagePost(MultiValueMap<String, String> paramMap) {

        // step 1. request
        String url = ViewModelConstants.HTTP_URL;
        // 初始化请求头并设置
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        // HttpEntity表示请求实体对象为request，泛型为MultiValueMap<String, String>，简单理解为可以同一个key下面放多个value
        HttpEntity< MultiValueMap<String, String> > request;
        // paramMap表示为存储图片的编码内容，headers则是请求头，封装到request请求实体中
        request = new HttpEntity<>(paramMap, headers);

        //step 2. post http call
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);


        // HttpStatus.OK：200
        Assert.isTrue(HttpStatus.OK == response.getStatusCode(), "http call is failed");
        return response.getBody();
    }
}
