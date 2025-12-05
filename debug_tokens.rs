use std::env;
use std::fs;

mod token;
mod lexer;

use lexer::Lexer;
use token::Token;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let filename = &args[1];
        let input = fs::read_to_string(filename).expect("Failed to read file");
        let mut lexer = Lexer::new(&input);
        
        loop {
            let tok = lexer.next_token();
            println!("{:?}", tok);
            if tok == Token::EOF {
                break;
            }
        }
    }
}
