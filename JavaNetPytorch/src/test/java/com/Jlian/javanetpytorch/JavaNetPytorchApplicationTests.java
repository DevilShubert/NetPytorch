package com.Jlian.javanetpytorch;

import com.Jlian.javanetpytorch.constant.RedisConstant;
import com.Jlian.javanetpytorch.utils.RedisUtils;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class JavaNetPytorchApplicationTests {
    @Autowired
    RedisUtils redisUtils;


    @Test
    void contextLoads() {
        Long firstUUID = redisUtils.listRightPush(RedisConstant.IMAGE_QUEUE_LIST, "firstUUID");
        System.out.println(firstUUID);
    }

}
