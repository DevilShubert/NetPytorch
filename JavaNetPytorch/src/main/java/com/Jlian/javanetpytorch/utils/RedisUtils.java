package com.Jlian.javanetpytorch.utils;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;
import org.springframework.util.CollectionUtils;

import java.util.Collection;
import java.util.Set;
import java.util.concurrent.TimeUnit;

@Component
public class RedisUtils {
    @Autowired
    @Qualifier("myRedisTemplateConfig")
    private RedisTemplate<String,Object> redisTemplate;

    /**
     * 指定缓存失效时间
     * @param key
     * @param time
     * @return boolean
     * @author JLian
     * @date 2021/11/19 10:26 下午
     */
    public boolean expire(String key,long time){
        try {
            if(time>0){
                redisTemplate.expire(key, time, TimeUnit.SECONDS);
            }
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
     * 根据key 获取过期时间
     * @param key
     * @return
     */
    public long getExpire(String key){
        return redisTemplate.getExpire(key, TimeUnit.SECONDS);
    }


    /**
     * 判断key是否存在
     * @param key 键
     * @return true 存在 false不存在
     */
    public boolean hasKey(String key){
        try {
            return redisTemplate.hasKey(key);
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
     * 删除缓存
     * @param key
     */
    @SuppressWarnings("unchecked")
    public void del(String ... key){
        if(key!=null&&key.length>0){
            if(key.length==1){
                redisTemplate.delete(key[0]);
            }else{
                redisTemplate.delete((Collection<String>) CollectionUtils.arrayToList(key));
            }
        }
    }


    //============================String=============================
    /**
     * 普通缓存获取
     * @param key 键
     * @return 值
     */
    public Object get(String key){
        return key==null ? null:redisTemplate.opsForValue().get(key);
    }

    /**
     * 普通缓存放入
     * @param key 键
     * @param value 值
     * @return true成功 false失败
     */
    public boolean set(String key,Object value) {
        try {
            redisTemplate.opsForValue().set(key, value);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
        * 获取所有key值
        * @return java.lang.String
        * @author JLian
        * @date 2021/11/21 3:03 下午
    */
    public String getAllKeys(){
        Set<String> keys = redisTemplate.keys("*");
        return keys.toArray().toString();
    }

    //============================List=============================
    /**
        * @param ListName
        * @return long
        * @author JLian
        * @date 2021/11/21 7:57 下午
    */
    public long getListLen(String ListName){
        Long size = redisTemplate.opsForList().size(ListName);
        return size;
    }

    /**
     * 队列入栈
        * @param listName  要插入的List
    	* @param listKeyValue  序列化后的list元素
        * @return java.lang.Long
        * @author JLian
        * @date 2021/11/21 8:01 下午
    */
    
    public Long listRightPush(String listName, String listKeyValue){
        Long rightPush = redisTemplate.opsForList().rightPush(listName, listKeyValue);
        return rightPush;
    }

    /**
     * 队列出栈
        * @param listName
        * @return java.lang.String
        * @author JLian
        * @date 2021/11/21 8:02 下午
    */

    public String listLeftPop(String listName){
        Object leftPop = redisTemplate.opsForList().leftPop(listName);
        return (String)leftPop;
    }


    //============================Hash=============================
    /**
     * @param HashName
     * @return java.lang.Long
     * @author JLian
     * @date 2021/11/21 7:58 下午
     */
    public Long getHashLen(String HashName){
        Long size = redisTemplate.opsForHash().size(HashName);
        return size;
    }

    /**
        * @param hashName  存放的redis缓存，用一个hash表来存储
    	* @param hashKey  hash的key值
    	* @param hashValue  hash的value值
        * @return void
        * @author JLian
        * @date 2021/11/21 8:06 下午
    */
    public void setHash(String hashName, String hashKey, String hashValue){
        redisTemplate.opsForHash().put(hashName, hashKey, hashValue);
    }

    /**
        * @param hashName
    	* @param hashKey
        * @return java.lang.Boolean
        * @author JLian
        * @date 2021/11/21 9:40 下午
    */
    public Boolean isHashKey(String hashName, String hashKey){
        Boolean hasKey = redisTemplate.opsForHash().hasKey(hashName, hashKey);
        return hasKey;
    }

    public String getHashValue(String hashName, String hashKey){
        Object CacheValue = redisTemplate.opsForHash().get(hashName, hashKey);
        return (String)CacheValue;
    }



}
