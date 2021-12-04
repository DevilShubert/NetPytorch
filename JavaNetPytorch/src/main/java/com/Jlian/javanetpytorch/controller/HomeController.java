package com.Jlian.javanetpytorch.controller;

import com.Jlian.javanetpytorch.constant.RedisConstant;
import com.Jlian.javanetpytorch.constant.ViewModelConstants;
import com.Jlian.javanetpytorch.service.ImageConsultService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.ModelAndView;

/**
 * Controller层的主页控制层
    * @return
    * @author JLian
    * @date 2021/11/16 9:50 下午
*/

@Controller
public class HomeController {

    @Autowired
    ImageConsultService imageConsultService;

    /**
        * detect index
        * @return java.lang.String
        * @author JLian
        * @date 2021/11/16 10:34 下午
    */

    @RequestMapping(value = {"/","/detect"})
    public String detect(){
        return ViewModelConstants.DETECT;
    }

    /**
     * 支持两种获取图片的方式，一种是直接选择上传本地的图片，另一种是数图输入文件的绝对路径
    	* @param imageFile
        * @return org.springframework.web.servlet.ModelAndView
        * @author JLian
        * @date 2021/11/17 10:04 上午
    */
    @RequestMapping(value = "/detectImage", method = RequestMethod.POST)
    public ModelAndView detectOut(MultipartFile imageFile) throws Exception {
        // 判断用户是直接上传本地图片还是上传的是图片绝对地址连接，并分别拿到目标检测后的图片
        String detectOut =  imageConsultService.consultByFile(imageFile);
        // 设置视图解析器
        ModelAndView modelAndView = new ModelAndView();
        modelAndView.setViewName(ViewModelConstants.DETECT_OUT);
        // 将String类型的检测结果直接作为对象返回到前端视图当中
        if (detectOut == RedisConstant.QUEUE_OVERFLOW_NOTICE) {
            modelAndView.addObject(ViewModelConstants.DETECT_OUT_NOTICE, detectOut);
            modelAndView.addObject(ViewModelConstants.DETECT_OUT_IMAGE, null);
        } else {
            modelAndView.addObject(ViewModelConstants.DETECT_OUT_IMAGE, detectOut);
        }
        return modelAndView;
    }
}


