import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiChatModelName;
import dev.langchain4j.model.output.structured.Description;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class CvClassifier {

    public record Person(
            String firstName,
            String lastName,
            String email,
            String website,
            String linkedin,
            @Description("A score from 0.0 to 10.0 regarding the job description")
            double score,
            @Description("Short text explaining the reason for the score")
            String scoreDescription) {
    }

    public record CandidatesWrapper(Person[] candidates) {}

    interface CvService{

        @UserMessage("""
                Compare the job description {{jobDescription}} with a list of resumes\s
                and return the list of candidates with a score : {{candidates}}
              """)
        CandidatesWrapper extractAndScore(@V("jobDescription")  String jobDescription,
                                          @V("candidates") List<String> candidates);
    }

    public static void main(String[] args) {
        Path cvsPath = Paths.get("src/main/resources/cvs");

        List<Document> documents = FileSystemDocumentLoader.loadDocuments(cvsPath);

        List<String> cvs = documents.stream().map(Document::text).toList();

        ChatLanguageModel model = OpenAiChatModel.builder().apiKey("OPENAI_API_KEY")
                .modelName(OpenAiChatModelName.GPT_4_O_MINI)
                .build();

        CvService cvService = AiServices.builder(CvService.class).chatLanguageModel(model).build();

        String jobDescription = "Software Engineer with 10+ years of experience in Java and Spring Boot";

        Person[] people = cvService.extractAndScore(jobDescription, cvs).candidates();

        for (Person p: people){
            System.out.println("----------------------------");

            System.out.printf("%s %s %n", p.firstName, p.lastName);
            System.out.printf("Email: %s, Site: %s, Linkedin: %s %n", p.email, p.website, p.linkedin);
            System.out.printf("Score: %s%n", p.score);
            System.out.println(p.scoreDescription);
        }
    }
}
