use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::fs::read_dir;
use path_slash::PathBufExt;

pub fn get_path_of_exec() -> String {
    let path = env::current_dir().unwrap();
    let s = path.to_slash().unwrap();
    let s1 = String::from(s);
    let path_of_exec = s1 + "/../";
    path_of_exec
}

pub fn get_path_to_config() -> String {
    let key = "config_path";
    let path_to_src = env::var(key).unwrap_or_else(|_e| {get_path_of_exec()});
    println!{"{}", path_to_src};
    path_to_src    
}

pub fn get_file_contents(fp: String) -> String {
    let mut file = File::open(fp.as_str()).unwrap();
    let mut contents = String::new();
    let res = file.read_to_string(&mut contents).unwrap();
    contents
}


pub fn get_all_files_in_directory(fp: String) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let it = read_dir(fp);
    for i in it.unwrap() {
        out.push(i.unwrap().file_name().into_string().unwrap());
    }
    out
}