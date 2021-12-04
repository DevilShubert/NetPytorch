package com.Jlian.javanetpytorch.detector.impl;

import com.Jlian.javanetpytorch.client.ImageDetecClient;
import com.Jlian.javanetpytorch.detector.ImageDetector;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;

@Component
public class ImageDetectorImpl implements ImageDetector {

    @Autowired
    ImageDetecClient imageDetecClient;

    /**
     * 编码图片
        * @param imageBase64Code
        * @return java.lang.String
        * @author JLian
        * @date 2021/11/16 9:59 下午
    */
    @Override
    public synchronized String getDetectImageByCode(String imageBase64Code) throws InterruptedException { // 加锁使代码串行执行
        // 模拟Python端检测图片过程，单个线程等待10s
        // Thread.currentThread().sleep(10000);
        LinkedMultiValueMap<String, String> paramMap = new LinkedMultiValueMap<>();
        paramMap.add("imageBase64Code", imageBase64Code);
        String detectImageOut = imageDetecClient.getDetectImagePost(paramMap);
        // 模拟从python端拿到了图片
        return detectImageOut;
    }
}
