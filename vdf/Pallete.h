#ifndef VDF_PALLETE_HEADER
#define VDF_PALLETE_HEADER
#include <map>
#include <set>
#include <vector>
#include <string>

class Pallete
{
	public:
		Pallete();
		/**
		 * @param id Color id to be returned
		 * @param opacity Opacity of color(0 transparent 15 opaque)
		 * @return String of the form #RGBA
		 */
		static std::string get(size_t id,int opacity);
		/**
		 * @param id Color id to be returned
		 * @param opacity Opacity of color(0 transparent)
		 * @return String of the form #RGBA
		 */
		static std::string get(std::string name,int opacity);
	private:
		static void init();
		static std::vector<std::string> colors;
		static std::set<size_t> unused_colors;
		static std::map<std::string,size_t> named_colors;
};
#endif
