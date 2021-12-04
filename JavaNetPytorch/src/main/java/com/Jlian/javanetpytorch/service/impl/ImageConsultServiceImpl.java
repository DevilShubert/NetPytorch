package com.Jlian.javanetpytorch.service.impl;

import com.Jlian.javanetpytorch.constant.RedisConstant;
import com.Jlian.javanetpytorch.detector.ImageDetector;
import com.Jlian.javanetpytorch.service.ImageConsultService;
import com.Jlian.javanetpytorch.utils.FileUtils;
import com.Jlian.javanetpytorch.utils.RedisUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import sun.misc.BASE64Encoder;

import java.io.*;
import java.util.UUID;


/**
 * 在这里检测Redis缓存层是否已经存在已有的UUID
 * 假如没有的话实现模拟队列，来限制最大访问的人数，首先判断是否达队列最大的size
 * 如果没有达到最大的size则先入队列然后等待python层返还数据之后，删除队列元素，在这个过程中调用client层方法
 */
@Service
public class ImageConsultServiceImpl implements ImageConsultService {


    @Autowired
    RedisUtils redisUtils;

    @Autowired
    ImageDetector imageDetector;

    @Autowired
    ImageQueueServiceImpl imageQueueService;


    /**
     *
        * @param imageFile - 图片文件
        * @return java.lang.String
        * @author JLian
        * @date 2021/11/21 2:28 下午
    */
    @Override
    public String consultByFile(MultipartFile imageFile) throws IOException, InterruptedException {
        if (imageFile.isEmpty()){
            return null;
        }
        BASE64Encoder base64Encoder = new BASE64Encoder();
        UUID uuidByFile = UUID.nameUUIDFromBytes(imageFile.getBytes());
        String base64EncoderImage = base64Encoder.encode(imageFile.getBytes());
        // 通过redisUtils来检测是否已经有这个对应的uuid
        if (!redisUtils.isHashKey(RedisConstant.CACHE_HASH, uuidByFile.toString())) {
            // 没有这个uuid说明还没有缓存，能检查list大小决定是否入队列
            if (redisUtils.getListLen(RedisConstant.IMAGE_QUEUE_LIST) >= RedisConstant.QUEUE_SIZE) {
                // 队列已经满了，提醒用户请重试
                return RedisConstant.QUEUE_OVERFLOW_NOTICE;
            } else {
                // 队列还未满，进行入队操作，之后拿到detect后的图片
                String imageDetectOut = imageQueueService.detectQueue(uuidByFile.toString(), base64EncoderImage);
                return imageDetectOut;
            }
        } else {
            // 对应的key已经存在，说明缓存命中并拿出对应的已经检测图像
            String consultResult = redisUtils.getHashValue(RedisConstant.CACHE_HASH, uuidByFile.toString());
            System.out.println("Cache Hits!");
            return consultResult;
        }
    }

}
