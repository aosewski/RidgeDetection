/**
 * @file graphDrawer.hpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is conducted under the supervision of prof. dr hab. inż. Marka
 * Nałęcza.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */
 
#ifndef GRAPHDRAWER_HPP_
#define GRAPHDRAWER_HPP_

#include "rd/utils/utilities.hpp"

#include <cstdio>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstddef>
#include <vector>
#include <map>
#include <iostream>

using std::size_t;

namespace rd 
{

template <typename T>
class GraphDrawer 
{
	
public:

	enum FIGURE
	{
		POINTS,
		SEGMENTS,
		LINE,
		CIRCLES,
		DATA_SECTIONS,
		END_DATA_SECTIONS,
	};

	// number of defined styles
	int stylesCnt;

	GraphDrawer() : currentGraphDim(0)
	{
	    // FILE *popen(const char *command, const char *mode);
	    // The popen() function shall execute the command specified by the string
	    // command, create a pipe between the calling program and the executed
	    // command, and return a pointer to a stream that can be used to either read
	    // from or write to the pipe.
		gnucmd = popen("gnuplot","w");

	    // popen() shall return a pointer to an open stream that can be used to read
	    // or write to the pipe.  Otherwise, it shall return a null pointer and may
	    // set errno to indicate the error.
	    if (!gnucmd) 
	    {
	        throw std::logic_error("Couldn't open connection to gnuplot");
	    }

	    std::string term = "set terminal pngcairo size 1280,800" 
	    		" enhanced notransparent font 'Verdana,12'";

	   	// for pdf output, only cm and inch are supported (size)
		// set terminal pdfcairo size 5.93cm,4.44cm enhanced font 'Verdana,10'
		// set output 'statistics.pdf'

		sendCmd(term);
		// sendCmd("set terminal postscript eps size 1280,800 "
		// 			"font 'Helvetica,20' linewidth 2");
		
		sendCmd("set key off");	 // bez legendy
		sendCmd("set size ratio -1");		// równe skale na osiach
		                                	
		// define axis
		// remove border on top and right and set color to gray
		sendCmd("set border 3 back lc rgb '#808080' lt 1.5");
		sendCmd("set tics nomirror");
		// sendCmd("set format '%g'");
		// define grid
		sendCmd("set grid back lc rgb '#808080' lt 0 lw 1");


		setDefaultPlotLineStyles();
	}

	virtual ~GraphDrawer() 
	{
		pclose(gnucmd);
	}

	//-------------------------------------------------
	// 	Automatic drawing functions with standard conf
	//-------------------------------------------------

	void showCircles(std::string filename,
					 T const *P,
					 int np,
					 T const *C,
					 int ns,
					 T r,
					 int dim = 2) const
	{

	    std::ostringstream cmdstr;

	    sendCmd("set size ratio -1");		// równe skale na osiach
		// cmdstr << "set out '" << filename << ".eps' ";
		cmdstr << "set out '" << filename << ".png' ";
		sendCmd(cmdstr.str());
	    cmdstr.clear();
	    cmdstr.str(std::string());
		
	    switch (dim) 
	    {
	    	case 2: cmdstr << "plot "; 
			    	if (P != nullptr) {
			    		cmdstr <<  "'-' w p pt 1 lc rgb '#d64f4f' ps 0.3, ";
			    	}
					cmdstr << "'-' w circles lc rgb 'blue' fs transparent solid 0.15 noborder, " <<
								"'-' u 1:2 w p pt 7 lc rgb 'black' ps 0.7";
		    	break;
	    	case 3: 
	    		// throw std::invalid_argument("Unsupported dimension!");
	    			cmdstr << "splot ";
	    			if (P != nullptr) {
			    		cmdstr << "'-' w p pt 1 lc rgb '#d64f4f' ps 0.3 ";
			    	}
			    	// cmdstr << "'-' w circles lc rgb 'blue' fs transparent solid 0.15 noborder, " <<
						  //   "'-' u 1:2 w p pt 7 lc rgb 'black'";
	    			break;

	    	default: throw std::invalid_argument("Unsupported dimension!");
	    }

		sendCmd(cmdstr.str());
		if (P != nullptr) 
		{
			printPoints(P, np, dim);
		}
		printCircles(C, ns, r, dim);
		printPoints(C, ns, dim);
	}	

	void showPoints(std::string filename,
					T const *points,
					int np,
					int dim = 2) const
	{
	    std::ostringstream cmdstr;

	    sendCmd("set size ratio -1");		// równe skale na osiach
		// cmdstr << "set out '" << filename << ".eps' ";
		cmdstr << "set out '" << filename << ".png' ";
	    sendCmd(cmdstr);

	    switch (dim) 
	    {
	    	case 2: cmdstr << "plot " << "'-' w p pt 2 lc rgb 'red' ps 0.7 ";
	    			break;
	    	case 3: /*sendCmd("set view 40, 45 ");*/
	    			cmdstr << "splot " << "'-' w p pt 2 lc rgb 'red' ps 0.7 ";
	    			break;
	    	default: throw std::invalid_argument("Unsupported dimension!");
	    }

		sendCmd(cmdstr.str());
		printPoints(points, np, dim);
		flushCmd();
	}

	/**
	 * @brief      Creates graph with distinct line segments.
	 *
	 * @param[in]  filename  Graph filename to write to.
	 * @param      points    Pointer to memory with data.
	 * @param[in]  nl        Number of lines to draw
	 * @param[in]  dim       Points dimension
	 */
	void showDistinctSegments(std::string filename, 
							T const *points,
							int nl,
							int dim = 2) const
	{
	    std::ostringstream cmdstr;

	    sendCmd("set size ratio -1");		// równe skale na osiach
		// cmdstr << "set out '" << filename << ".eps' ";
		cmdstr << "set out '" << filename << ".png' ";
		sendCmd(cmdstr.str());
	    cmdstr.clear();
	    cmdstr.str(std::string());


	    switch (dim) 
	    {
	    	case 2: cmdstr << "plot " << "'-' w l lt 2 lc rgb 'red' lw 2 ";
	    			break;
	    	case 3: /*sendCmd("set view 40, 45 ");*/
	    			cmdstr << "splot " << "'-' w l lt 2 lc rgb 'red' lw 2, ";
	    			break;
	    	default: throw std::invalid_argument("Unsupported dimension!");
	    }

		sendCmd(cmdstr.str());
		printLines(points, nl, dim);
		flushCmd();
	}

	//-------------------------------------------------
	// 	Graph customization
	//-------------------------------------------------

	void setXLabel(std::string label) const
	{
		sendCmd(std::string("set xlabel '") + label + std::string("'"));
	}

	void setYLabel(std::string label) const
	{
		sendCmd(std::string("set ylabel '") + label + std::string("'"));
	}

	/**
	 * @brief      Shows legend in standard position (bottom right)
	 */
	void showLegend() const
	{
		sendCmd("set key bottom right");
	}

	size_t defineLineStyle(std::string style)
	{
		sendCmd(std::string("set style line ") + std::to_string(++stylesCnt) + std::string(" ") + style);
		return stylesCnt;
	}

	/**
	 * @brief set path to config files directory
	 */
	void setLoadPath(std::string dirPath) const
	{
		std::string cfgFilesDirPath = findPath(dirPath, "", true);
		sendCmd("set loadpath '" + cfgFilesDirPath + "'");
	}

	void loadConfigFile(std::string fileName) const
	{
		sendCmd("load '" + fileName + "'");
	}

	/**
	 * @brief      Custom 3D graph configuration
	 */
	void setGraph3DConf() const 
	{
        // set borders without front and front top lines
        sendCmd("set border 127+256+512 back lc rgb '#808080' lt 1.5");
        sendCmd("set xtics border offset -0.7,-0.5,0; set ytics border"
            " offset 2.5,0,0; set ztics border;");
        sendCmd("set grid xtics ytics ztics back lc rgb '#808080' lt 0 lw 1");
        // set xyplane exactly at z_min. given value specify fraction of z-range 
        sendCmd("set xyplane relative 0");
        // x-rot, z-rot, scale
        sendCmd("set view 65, 30, 1.15");
	}

	//-------------------------------------------------
	// 	API for incremental adding graph data
	//-------------------------------------------------
	void startGraph(std::string const &filename, int dim=2)
	{
		clean();
	    std::ostringstream cmdstr;
		cmdstr << "set out '" << filename << ".png' ";
		sendCmd(cmdstr);
		currentGraphDim = dim;
		if (dim <= 2)
		{
			plotCommand_ << "plot ";
		}
		else if (dim == 3)
		{
			plotCommand_ << "splot ";
		}
		else if (dim > 3)
		{
			throw std::logic_error("Can't draw graph with dim > 3!");
		}
	}
	
	void endGraph()
	{
		sendCmd(plotCommand_);
		printDrawingData(currentGraphDim);
		flushCmd();
		currentGraphDim = 0;
	}

	/**
	* @brief      Adds drawing comand arguments to current graph.
	 *
	 * @param[in]  cmd      Argument string.
	 * @param[in]  data     Pointer to data.
	 * @param[in]  figType  Enumeration - figure to draw.
	 * @param[in]  size     Number of elements to draw.
	 */
	void addPlotCmd(std::string const &cmd,
					 T const * data,
					 FIGURE figType, 
					 int size)
	{
		plotCommand_ << cmd << ", ";
		plotData_.push_back(data);
		plotDataSize_.push_back(size);
		plotDataType_.push_back(figType);
	}

	void addPlotCmd(std::string const &cmd,
					 std::vector<std::vector<T>> const &data,
					 FIGURE figType)
	{
		plotCommand_ << cmd << ", ";
		for (auto const &v : data)
		{
			plotData_.push_back(v.data());
			plotDataSize_.push_back(v.size() / currentGraphDim);
			plotDataType_.push_back(figType);
		}
		plotDataType_.push_back(END_DATA_SECTIONS);
	}


	/**
	 * @brief      Sends custom command to gnuplot
	 *
	 * @param[in]  cmdstr  Command to execute
	 */
	void sendCmd(const std::string &cmdstr) const
	{
	    // int fputs ( const char * str, FILE * stream );
	    // writes the string str to the stream.
	    // The function begins copying from the address specified (str) until it
	    // reaches the terminating null character ('\0'). This final
	    // null-character is not copied to the stream.
		// std::cout << "gnuplot: " << cmdstr << std::endl;

		// #ifdef RD_DEBUG
		// 	std::cout << ">>>> sendCmd: " << cmdstr << "\n";
		// #endif

	    fputs((cmdstr+"\n").c_str(), gnucmd);
	}


private:

	int currentGraphDim;
	std::ostringstream 				plotCommand_;
	std::vector<T const*> 			plotData_;
	std::vector<int> 				plotDataSize_;
	std::vector<FIGURE> 			plotDataType_;
    ///@brief pointer to the stream that can be used to write to the pipe
    FILE 					*gnucmd;

    /**
     * @brief      Creates default line styles from spectral.pal palette
     */
    void setDefaultPlotLineStyles()
    {
    	sendCmd("set style line 1 pt 1 ps 1 lt 1 lw 2 lc rgb '#D53E4F'"); // red
		sendCmd("set style line 2 pt 2 ps 1 lt 1 lw 2 lc rgb '#F46D43'"); // orange
		sendCmd("set style line 3 pt 3 ps 1 lt 1 lw 2 lc rgb '#FDAE61'"); // pale orange
		sendCmd("set style line 4 pt 4 ps 1 lt 1 lw 2 lc rgb '#FEE08B'"); // pale yellow-orange
		sendCmd("set style line 5 pt 5 ps 1 lt 1 lw 2 lc rgb '#E6F598'"); // pale yellow-green
		sendCmd("set style line 6 pt 6 ps 1 lt 1 lw 2 lc rgb '#ABDDA4'"); // pale green
		sendCmd("set style line 7 pt 7 ps 1 lt 1 lw 2 lc rgb '#66C2A5'"); // green
		sendCmd("set style line 8 pt 8 ps 1 lt 1 lw 2 lc rgb '#3288BD'"); // blue
		stylesCnt = 8;		                                                                 
    }

	/**
	 * @brief      Clears drawing resources.
	 */
	void clean()
	{
		plotCommand_.clear();
		plotCommand_.str(std::string());
		plotData_.clear();
		plotDataSize_.clear();
		plotDataType_.clear();
	}

	void printDrawingData(int dim = 2) const
	{
		for (size_t k = 0; k < plotDataType_.size(); ++k)
		{
			switch (plotDataType_[k])
			{
				case POINTS:
					printPoints(plotData_[k], plotDataSize_[k], dim);
					break;
				case SEGMENTS:
					printSegments(plotData_[k], plotDataSize_[k], dim);
					break;
				case LINE:
					printPoints(plotData_[k], plotDataSize_[k], dim);
					break;
				case CIRCLES:
					printCircles(plotData_[k], plotDataSize_[k], dim);
					break;
				case DATA_SECTIONS:
					printPoints(plotData_[k], plotDataSize_[k], dim, true);
					break;
				case END_DATA_SECTIONS:
					// end of input data
					sendCmd("e");
					break;
				default:
					throw std::invalid_argument("Unsupported plot data type!");
			}
		}
	}

	void printSegments(T const *points, int size, int dim = 2) const
	{
		int i = 0;
		std::ostringstream cmdstr;
		while (size-- > 0)
		{
			while (i++ < dim)
			{
				checkIsNaN(points);
				cmdstr << std::right << std::fixed << std::setw(10) << std::setprecision(4) << 
						*(points++) << " ";
			}
			i = 0;
			cmdstr << "\n ";
			while (i++ < dim)
			{
				checkIsNaN(points);
				cmdstr << std::right << std::fixed << std::setw(10) << std::setprecision(4) << 
						*(points++) << " ";
			}
			// "empty data line to start drawing new line"
			cmdstr << "\n \n";
			i = 0;
		    sendCmd(cmdstr);
		}
		// end of input data
		sendCmd("e");
	}

	void printPoints(T const *points, int size, int dim = 2, bool isMoreData = false) const
	{
		int i = 0;
		std::ostringstream cmdstr;
		while (size-- > 0) 
		{
	    	while (i++ < dim) 
	    	{
	    		checkIsNaN(points);
		    	cmdstr << std::right << std::fixed << std::setw(10) << std::setprecision(4) << 
		    	*(points++) << " ";
			}
			// add additional y coordinate value equal to zero
			if (dim == 1)
			{
				cmdstr << 0 << " ";
			}
		    i = 0;
		    sendCmd(cmdstr);
		}
		if (isMoreData)
		{	
			// two empty lines to mark data section end (reset gnuplot pseudocolumn(0) aka $0 counter)
		    sendCmd("\n");
		}
		else
		{
			// end of input data
			sendCmd("e");
		}
	}

	void printCircles(T const *points, int size, T r, int dim = 2) const
	{
		int i=0;
		std::ostringstream cmdstr;
		while (size-- > 0) 
		{
	    	while (i++ < dim) 
	    	{
	    		checkIsNaN(points);
		    	cmdstr << std::right << std::fixed << std::setw(10) << std::setprecision(4) << 
		    	*(points++) << " ";
			}
			cmdstr << std::right << std::fixed << std::setw(10) << std::setprecision(4) << r << " ";
		    i = 0;
		    sendCmd(cmdstr);
		}
		sendCmd("e");
	}
	
	void printCircles(T const *points, int size, int dim = 2) const
	{
		int i=0;
		std::ostringstream cmdstr;
		while (size-- > 0) 
		{
			// last number is a circle radius
	    	while (i++ < dim + 1) 
	    	{
	    		checkIsNaN(points);
		    	cmdstr << std::right << std::fixed << std::setw(10) << std::setprecision(4) << 
		    	*(points++) << " ";
			}
		    i = 0;
		    sendCmd(cmdstr);
		}
		sendCmd("e");
	}

	/**
	 * @brief      Flushes command to gnuplot.
	 *
	 * @param      s     Stream object containing command string.
	 */
	void sendCmd(std::ostringstream &s) const
	{
		sendCmd(s.str());
		s.clear();
		s.str(std::string());
	}

	void flushCmd() const
	{
	    // int fflush ( FILE * stream );
	    // If the given stream was open for writing and the last i/o operation was
	    // an output operation, any unwritten data in the output buffer is written
	    // to the file.  If the argument is a null pointer, all open files are
	    // flushed.  The stream remains open after this call.
	    fflush(gnucmd);
	}

	void checkIsNaN(T const *value) const
	{
		if (std::isnan(*value)) 
		{
			throw std::logic_error("Invalid Data (NaN)!");
		}
	}

};

} // end namespace rd

#endif /* GRAPHDRAWER_HPP_ */
