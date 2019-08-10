
/*
   Creates randomize sets of a chemical reaction. All reactions are written in a file in the following format:
   reactant1.reactant2.reactant3>>product 

   For example, for etherification reaction the line can be:
   c1ccccc1CO.CO>>c1ccccc1COC
   
   And the output 

   CO.c1(CO)ccccc1>>COCc1ccccc1
   CO.c1c(CO)cccc1>>COCc1ccccc1
   CO.c1cc(ccc1)CO>>COCc1ccccc1
   OC.OCc1ccccc1>>COCc1ccccc1
   OCc1ccccc1.CO>>COCc1ccccc1
   OCc1ccccc1.OC>>COCc1ccccc1
   c1c(CO)cccc1.CO>>COCc1ccccc1
   c1cc(CO)ccc1.OC>>COCc1ccccc1
   c1cccc(c1)CO.CO>>COCc1ccccc1
 
   All reagents are shuffled, and also all reactants SMILES are written in random anti-canonical order.
   Maximum number of variants is controlled by RND parameter.

*/

#include <openbabel/obconversion.h>
#include <openbabel/mol.h>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <list>
#include <stdlib.h>
#include <time.h>
#include <set>

const int RND = 10;

int main(int argc, char **argv)
{
    if(argc != 2)
    {
         printf("Usage: %s data-file.\n", argv[0]);
         return EXIT_SUCCESS;
    }

    std::ifstream ifs;
    ifs.open(argv[1], std::ifstream::in);
   
    if(!ifs.good())
    {
         fprintf(stderr, "Problem with the file: %s\n", argv[1]);
         return EXIT_FAILURE;
    }
    
    //for reagents we use anti-canonical
    OpenBabel::OBConversion conv;
    conv.SetInAndOutFormats("SMI", "SMI"); 
    conv.SetOutStream(&std::cout);
    conv.SetOptions("Cn", OpenBabel::OBConversion::OUTOPTIONS);

    //but for products canonical
    OpenBabel::OBConversion conv_c;
    conv_c.SetInAndOutFormats("SMI", "SMI"); 
    conv_c.SetOutStream(&std::cout);
    conv_c.SetOptions("cn", OpenBabel::OBConversion::OUTOPTIONS);

    srand(time(NULL));

    int lines = 0;
    while(true)
    {
        lines++;        

        std::string line;
        getline(ifs, line);
        if(ifs.eof())
           break;

        std::vector<std::string> items;
        boost::algorithm::split(items, line, boost::algorithm::is_any_of(">"));
     
        //reactants>reagents>products
        if(items.size() != 1)
        {
             fprintf(stderr, "Erron on line: %d.\n", lines);
             return EXIT_FAILURE;
        }        
 
        std::vector<std::string> compounds;
        boost::algorithm::split(compounds, items[0], boost::algorithm::is_any_of("."));
  
        std::vector<OpenBabel::OBMol> reactants;
        std::vector<OpenBabel::OBMol> products;  

        //reactants
        OpenBabel::OBMol mol; 
        for(int i=0; i< compounds.size(); ++i)
        {
            std::stringstream in;
            in.str(compounds[i]);
            
            conv.SetInStream(&in);
            if(!conv.Read(&mol))
	    {
                 fprintf(stderr, "Error reading a molecule on line: %d!\n", lines);
                 return EXIT_FAILURE;
            }
             
            reactants.push_back(mol);
        }
     
        //products
        boost::algorithm::split(compounds, items[0], boost::algorithm::is_any_of(":"));
        for(int i=0; i< compounds.size(); ++i)
        {
            std::stringstream in;
            in.str(compounds[i]);
           
            conv.SetInStream(&in);
            if(!conv.Read(&mol))
	    {
                 fprintf(stderr, "Error reading a molecule on line: %d!\n", lines);
                 return EXIT_FAILURE;
            }
             
            products.push_back(mol);
        }

        if(products.size() != 1)
        {
            fprintf(stderr, "More than one product found on line %d.\n", lines);
            return EXIT_FAILURE;
        }
  
        //print product in canonical way 
        std::stringstream prod;
        conv_c.SetOutStream(&prod);
        if(!conv_c.Write(&products[0]))
        {
            printf("Error writing the product on line %d.\n", lines);
            return EXIT_FAILURE;
        }

        std::string pmol = prod.str();
        pmol.erase(pmol.length() - 1);

        std::set <std::string> offer;
        for(int rnd = 0; rnd < RND; rnd++)
        {
            const int reactants_len = reactants.size();
            std::vector<int> inds(reactants_len, -1);
           
            for(int i=0; i< reactants_len; i++)
            {               
                while(true)
                {
                   int pos = rand() % reactants_len;
                   int found = 0;
                   for(int j=0; j< i; j++)
                      if(pos == inds[j]) 
                      {
                          found = 1;
                          break;
                      }

                   if(!found)
                   {
                       inds[i] = pos;
                       break;
                   }
                }
            }

            std::vector<std::string> left(reactants.size());
            std::vector<std::string> right(products.size());

            for(int i=0; i< reactants.size(); ++i)
            {      
                 std::stringstream out;

                 conv.SetOutStream(&out);
                 if(!conv.Write(&reactants[i]))
                 {
                     fprintf(stderr, "Error writing molecule on line: %d!\n", lines);
                     return EXIT_FAILURE;
                 }
                 
                 std::string nmol = out.str();
                 nmol.erase(nmol.length() - 1);
                 left[inds[i]] = nmol;                 
             }
 
             std::stringstream res_left;
             for(int i=0; i< reactants_len-1; i++)
                 res_left << left[i] << ".";
             res_left << left[reactants_len-1] << ">>" << pmol; 

             offer.insert(res_left.str());
        }            

        for(std::set<std::string>::const_iterator it = offer.begin(); it != offer.end(); ++it)
            printf("%s\n", (*it).c_str());       
        
    }	


    ifs.close();

    return EXIT_SUCCESS;
}

