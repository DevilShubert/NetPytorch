package com.Jlian.javanetpytorch.service.impl;

import com.Jlian.javanetpytorch.constant.RedisConstant;
import com.Jlian.javanetpytorch.detector.ImageDetector;
import com.Jlian.javanetpytorch.service.ImageQueueService;
import com.Jlian.javanetpytorch.utils.RedisUtils;
import com.alibaba.fastjson.JSON;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
@Async
public class ImageQueueServiceImpl implements ImageQueueService {

    @Autowired
    RedisUtils redisUtils;

    @Autowired
    ImageDetector imageDetector;

    /**
     *
        * @param uuid - 准备入队的hashKey
    	* @param base64EncoderImage 准备入队列的hashValue
        * @return java.lang.String
        * @author JLian
        * @date 2021/11/21 7:51 下午
    */

    @Override
    public String detectQueue(String uuid, String base64EncoderImage) throws InterruptedException {

        // 首先将uuid和base64EncoderImage入队列
        Map<String, Object> imageInfo = new HashMap<>();
        imageInfo.put(RedisConstant.QUEUE_LIST_KEY, uuid);
        imageInfo.put(RedisConstant.QUEUE_LIST_VALUE, base64EncoderImage);
        redisUtils.listRightPush(RedisConstant.IMAGE_QUEUE_LIST, JSON.toJSONString(imageInfo));

        // 没有这个对应的key，直接拿到下一层去detect
        String detectImageByCode = imageDetector.getDetectImageByCode(base64EncoderImage);


        // 出栈，腾出空间
        redisUtils.listLeftPop(RedisConstant.IMAGE_QUEUE_LIST);

        // 保存
        redisUtils.setHash(RedisConstant.CACHE_HASH, uuid, detectImageByCode);

        return detectImageByCode;
    }
}
