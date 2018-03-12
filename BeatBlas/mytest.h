#include <string>
#include <iostream>


  static const std::string NAME = "MyTest";

class MyTest{
public:
 enum class Debug_Level{FULL, PARTIAL, BRIEF, NONE};
 enum class ALGORITHM{add, subtract, multiply};


private:

 Debug_Level debug_level;
 int x, y;

 void reset(){
 x = 0;
 y = 0;
}

public:

 void set_x(int x_){
	switch((int)debug_level) {
	 case 0: std::cout << "method: void set_x(int)" << std::endl;
	 case 1: std::cout <<  "set_x param ";
	 case 2: std::cout <<  "x_ = " << x_ << std::endl;
	 case 3: break;
	}
 x = x_;
 }

 void set_y(int y_){
switch((int)debug_level) {
	 case 0: std::cout << "method: void set_y(int)" << std::endl;
	 case 1: std::cout <<  "set_y param ";
	 case 2: std::cout <<  "y_ = " << y_ << std::endl;
	 case 3: break;
	}
 y = y_;
 }

 const std::string& name(){
   return NAME;
 }

 void set_debug(Debug_Level current_level){
   debug_level = current_level;
 }

 void debug() {
 std::cout <<  "X = " << x << std::endl;
 std::cout <<  "Y = " << y << std::endl;
}

 int compute(ALGORITHM input) {

 if(input == ALGORITHM::add) return compute_1();
 else if(input == ALGORITHM::subtract) return compute_2();
 else return compute_3();

}

private:
int compute_1(){
switch((int)debug_level) {
	 case 0: std::cout << "method: int compute_1()" << std::endl;
	 case 1: std::cout <<  "(add) ";
	 case 2: std::cout <<  "sum = " << x+y << std::endl;
	 case 3: break;
	}
return x + y;
}

int compute_2(){
switch((int)debug_level) {
	 case 0: std::cout << "method: int compute_2()" << std::endl;
	 case 1: std::cout <<  "(subtract) ";
	 case 2: std::cout <<  "difference = " << x-y << std::endl;
	 case 3: break;
	}
return x - y;
}

int compute_3(){
switch((int)debug_level) {
	 case 0: std::cout << "method: int compute_3()" << std::endl;
	 case 1: std::cout <<  "(multiplies) ";
	 case 2: std::cout <<  "product = " << x*y << std::endl;
	 case 3: break;
	}
return x*y;
}

};

