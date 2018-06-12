#ifndef BC_ONLINE_PRINTER_
#define BC_ONLINE_PRINTER_

#include <type_traits>
#include <iostream>

namespace BC {

  struct NewLine {} ;
  struct PrintContinue {};
  static constexpr NewLine nl = NewLine();
  static constexpr PrintContinue printC = PrintContinue();

  struct Printer {

    const PrintContinue& operator - (std::string param) const {
      std::cout << param;
      return printC;
    }
    template<class T>
    const PrintContinue& operator - (T param) const {
      std::cout << param;
      return printC;
    }
    const PrintContinue& operator - (const NewLine& param) const {
      std::cout << std::endl;
      return printC;
    }

  };

  static constexpr Printer print = Printer();

    const PrintContinue& operator , (const PrintContinue& pc, std::string param)  {
      std::cout << param;
      return printC;
    }
    const PrintContinue& operator , (const PrintContinue& pc, const NewLine& param)  {
      std::cout << std::endl;
      return printC;
    }
    const PrintContinue& operator - (const PrintContinue& pc, const NewLine& param)  {
      std::cout << std::endl;
      return printC;
    }
}

#endif
