// 提交时的验证
function toValid() {
    let inputGroupFile = document.getElementById("inputGroupFile").value;
    if (isEmpty(inputGroupFile) ) {
        alert("Picture file is empty!");
        return false;
    }
}

// 检测是否为空的资源路径或者文件
function isEmpty(str) {
    if (typeof str == null || str == "" || str == "undefined") {
        return true;
    } else {
        if (str.match(/^[ ]*$/)) {
            return true;
        }
        return false;
    }
}

function changeImageFile() {
    let selectedFile = document.getElementById('inputGroupFile').files[0];
    let name = selectedFile.name;//读取选中文件的文件名
    document.getElementById("inputFileLabelId").innerHTML = name;
    document.getElementById("imageLinkId").value = '';

}


// 检测不同操作系统下是否为有效资源
function isValidUrl(imageLinkId) {
    if (isEmpty(imageLinkId)) return false;
    let winPath = new RegExp("^[A-z]:\\\\(.+?\\\\)*$");
    let LinuxPath = new RegExp(/\/([\w\.]+\/?)*/);

    if (winPath.test(imageLinkId) || LinuxPath.test(imageLinkId)) {
        if (/\.(gif|jpg|jpeg|png|GIF|JPEG|JPG|PNG)$/.test(imageLinkId)) {
            return true;
        } else {
            alert("All uploaded files must be [.jpeg, .jpg, .png] type pictures!");
            return false;
        }
    } else {
        alert("this link is not correct!")
        return false;
    }
}
function changeImageLink() {
    document.getElementById("inputGroupFile").value=null;
    document.getElementById("inputFileLabelId").innerHTML = 'Choose file';
}