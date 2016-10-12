import com.sun.jna.Structure;
import java.util.List;
import java.util.Arrays;

public class DarkGrayArgs extends Structure
{
	public int width;
	public int height;
	protected List<String> getFieldOrder()
	{
		return Arrays.asList(new String[] { "width", "height"});
	}

	public String toString()
	{
		return width+" * "+height;
	}
}
